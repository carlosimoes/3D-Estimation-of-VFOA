"""Command line interface: ``python -m vfoa run ...``.

``run`` builds the configuration (profile <- config file <- ``--set`` <- flags),
opens the frame source, streams it through :class:`vfoa.pipeline.Pipeline`
and writes into ``output.dir``:

* ``config.yaml``   the resolved configuration (written before the run),
* ``results.jsonl`` one JSON object per frame (``output.jsonl``),
* ``results.csv``   one row per (frame, person, object) (``output.csv``),
* ``overlay.mp4``   overlay video (``output.video`` / ``--video``),
* ``timing.json``   the run summary (per-stage timing, cache statistics; in
  real-time mode also ``counters.dropped_frames``/``counters.captured_frames``,
  ``latency`` capture->result statistics in ms, ``async`` worker statistics of
  asynchronous stages and ``interrupted: true`` after Ctrl-C),

and prints the per-stage timing table. ``--realtime`` wraps the source in a
latest-frame reader that drops frames the pipeline cannot keep up with;
``--set segmentation.async=true`` runs segmentation on a worker thread. Ctrl-C
stops an endless stream cleanly: the summary is still written and the exit
code is 130. Re-scoring without models: run the same
command with ``--set cache.enabled=true`` once, then again with
``--set cache.enabled=true --set cache.mode=read`` plus new attention or
smoothing settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = ["build_parser", "config_from_args", "build_sinks", "live_lines", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``vfoa`` command."""
    parser = argparse.ArgumentParser(prog="vfoa", description="3D Visual Focus of Attention pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="process a video, image directory, camera or the synthetic world")
    run.add_argument("--config", default=None, help="YAML configuration file")
    run.add_argument("--profile", default=None, help="profile: fast|accurate|thesis|synthetic")
    run.add_argument("--input", default=None,
                     help="video file, image directory, camera index or 'synthetic' (input.path)")
    run.add_argument("--right", default=None, help="rectified right view (input.right_path)")
    run.add_argument("--camera-dir", default=None, help="Sintel .cam directory (input.camera_dir)")
    run.add_argument("--depth-dir", default=None, help="Sintel .dpt directory (input.depth_dir)")
    run.add_argument("--output-dir", default=None, help="output directory (output.dir)")
    run.add_argument("--max-frames", type=int, default=None, help="process at most N frames")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="config override, e.g. --set attention.sigma_gaze_deg=5 (repeatable)")
    run.add_argument("--realtime", action="store_true",
                     help="drop frames to keep up with the source (input.realtime)")
    run.add_argument("--video", action="store_true", help="write overlay.mp4 (output.video)")
    run.add_argument("--show", action="store_true", help="show the overlay in a window (output.show)")
    run.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def config_from_args(args: argparse.Namespace) -> Any:
    """Resolved :class:`vfoa.config.Config` of the parsed ``run`` arguments."""
    from vfoa.config import load_config  # noqa: PLC0415

    sets = list(args.set or [])
    cfg = load_config(args.config, args.profile, overrides=sets or None)
    if args.input is not None:
        cfg.input.path = str(args.input)
        if not any(s.split("=", 1)[0].strip() == "input.source" for s in sets):
            cfg.input.source = "auto"
    if args.right is not None:
        cfg.input.right_path = str(args.right)
    if args.camera_dir is not None:
        cfg.input.camera_dir = str(args.camera_dir)
    if args.depth_dir is not None:
        cfg.input.depth_dir = str(args.depth_dir)
    if args.max_frames is not None:
        cfg.input.max_frames = int(args.max_frames)
    if args.output_dir is not None:
        cfg.output.dir = str(args.output_dir)
    if args.realtime:
        cfg.input.realtime = True
    if args.video:
        cfg.output.video = True
    if args.show:
        cfg.output.show = True
    return cfg


def build_sinks(cfg: Any, source: Any) -> list:
    """Output sinks enabled in ``cfg.output`` (files go to ``output.dir``)."""
    from vfoa.io.outputs import CsvSink, DisplaySink, JsonlSink, VideoSink  # noqa: PLC0415

    out = cfg.get("output") or {}
    out_dir = Path(out.get("dir") or "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sinks: list = []
    try:
        if out.get("jsonl", True):
            sinks.append(JsonlSink(out_dir / "results.jsonl"))
        if out.get("csv"):
            sinks.append(CsvSink(out_dir / "results.csv"))
        if out.get("video"):
            sinks.append(VideoSink(out_dir / "overlay.mp4", getattr(source, "fps", None) or 30.0))
        if out.get("show"):
            sinks.append(DisplaySink())
    except Exception:
        for s in sinks:
            s.close()
        raise
    return sinks


def live_lines(summary: Any) -> list[str]:
    """Human-readable latency / asynchronous-stage lines of a run summary."""
    lines: list[str] = []
    lat = summary.get("latency") or {}
    if lat:
        lines.append(
            f"latency capture->result: mean {lat['mean_ms']:.1f} ms  p50 {lat['p50_ms']:.1f}  "
            f"p95 {lat['p95_ms']:.1f}  max {lat['max_ms']:.1f}  ({lat['frames']} frames)"
        )
    for stage, st in (summary.get("async") or {}).items():
        lines.append(
            f"async {stage}: submitted {st['submitted']}  skipped {st['skipped']}  completed {st['completed']}  "
            f"failed {st['failed']}  worker mean {st['mean_ms']:.1f} ms  max {st['max_ms']:.1f} ms"
        )
    return lines


def _run(args: argparse.Namespace) -> int:
    from vfoa.config import dump_config  # noqa: PLC0415
    from vfoa.pipeline import Pipeline, open_input  # noqa: PLC0415

    cfg = config_from_args(args)
    out_dir = Path(cfg.output.get("dir") or "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")
    source = open_input(cfg)
    try:
        sinks = build_sinks(cfg, source)
    except Exception:
        source.close()
        raise
    pipeline = Pipeline(cfg)
    interrupted = False
    try:
        summary = pipeline.run(source, sinks)
    except KeyboardInterrupt:
        interrupted = True
        summary = dict(pipeline.last_summary)
        summary["interrupted"] = True
    with open(out_dir / "timing.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, default=str)
    print(pipeline.timer.report())
    for line in live_lines(summary):
        print(line)
    print(f"results written to {out_dir}" + (" (interrupted)" if interrupted else ""))
    return 130 if interrupted else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code (2 on configuration/input errors)."""
    from vfoa.backends.base import BackendUnavailable  # noqa: PLC0415

    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "run":
            return _run(args)
    except (ValueError, FileNotFoundError, BackendUnavailable) as exc:
        print(f"vfoa: error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
