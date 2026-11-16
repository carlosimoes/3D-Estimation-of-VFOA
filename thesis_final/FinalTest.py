"""Legacy thesis entry point: ``python thesis_final/FinalTest.py [flags]``.

Runs :class:`VideoProcessor` (the streaming vfoa pipeline) on a video. The
legacy flags are kept and mapped onto the vfoa configuration (default profile
``thesis``: the thesis models with the bug-fixed thesis algorithm); model
paths default to the config values, which are relative to
``models.third_party_root`` / ``$VFOA_THIRD_PARTY``. New flags: ``--config``,
``--profile``, ``--set a.b=c`` (repeatable) and ``--max-frames``.

Outputs land in ``<-od or output.dir>/<input stem>/`` (results.jsonl,
results.csv, timing.json, config.yaml, features.desc, feature videos).
Nothing runs at import time.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

__all__ = ["build_parser", "main"]


def _str2bool(value: object) -> bool:
    """Real boolean parsing of a command-line value (true/false, yes/no, 1/0, on/off)."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Parser of the legacy flags plus ``--config/--profile/--set/--max-frames``."""
    p = argparse.ArgumentParser(
        prog="FinalTest.py",
        description="3D Visual Focus of Attention on a video (legacy thesis entry point)",
    )
    p.add_argument("opts", nargs="*", default=[], help="ignored (legacy)")
    # new flags
    p.add_argument("--config", default=None, help="YAML configuration file")
    p.add_argument("--profile", default=None,
                   help="profile: fast|accurate|thesis|synthetic (default: the config file's, else thesis)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set attention.method=angular (repeatable)")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="process at most N frames")
    # inputs
    p.add_argument("-d", default=None, help="path of the (rectified) video to analyse")
    p.add_argument("-dvL", default=None, help="left stereo view (video or image directory)")
    p.add_argument("-dvR", default=None, help="right stereo view (video or image directory)")
    p.add_argument("-dvC", default=None, help="directory of the Sintel .cam files of the left camera")
    p.add_argument("-od", default=None, help="output directory (outputs go to <od>/<input stem>/)")
    # behaviour
    p.add_argument("-w", type=_str2bool, nargs="?", const=True, default=True,
                   help="write the feature videos (true/false)")
    p.add_argument("-g", type=_str2bool, nargs="?", const=True, default=True,
                   help="estimate head pose and gaze (true/false)")
    p.add_argument("-m", type=_str2bool, nargs="?", const=True, default=False,
                   help="match keypoints and head boxes (true/false)")
    p.add_argument("-dev", default=None,
                   help="device: cpu, cuda, cuda:N, or GPU indices for CUDA_VISIBLE_DEVICES (e.g. 0,1)")
    p.add_argument("-b", type=int, default=20000, help="legacy batch size (ignored: the video is streamed)")
    # thesis model paths/thresholds (default: config values relative to models.third_party_root)
    p.add_argument("-cfg", default=None, help="OpenPose experiment config (persons.openpose_cfg)")
    p.add_argument("-opM", default=None, help="OpenPose weights (persons.openpose_weights)")
    p.add_argument("-dpCfg", default=None, help="DensePose config (persons.densepose_cfg)")
    p.add_argument("-dpM", default=None, help="DensePose weights (persons.densepose_weights)")
    p.add_argument("-PpCfg", default=None, help="Panoptic FPN config (segmentation.panoptic_cfg)")
    p.add_argument("-PpM", default=None, help="Panoptic FPN weights (segmentation.panoptic_weights)")
    p.add_argument("-gM", default=None, help="Gaze360 weights (gaze.weights)")
    p.add_argument("-PpB", type=int, default=None, help="ignored (legacy batch size)")
    p.add_argument("-opB", type=int, default=None, help="ignored (legacy batch size)")
    p.add_argument("-dpB", type=int, default=None, help="ignored (legacy batch size)")
    p.add_argument("-gB", type=int, default=None, help="Gaze360 batch size (gaze.batch_size)")
    p.add_argument("-PpT", type=float, default=None, help="Panoptic score threshold (segmentation.panoptic_thr)")
    p.add_argument("-dpT", type=float, default=None, help="DensePose score threshold (persons.densepose_thr)")
    p.add_argument("-gT", type=float, default=None, help="Gaze360 uncertainty threshold (gaze.uncertainty_thr)")
    p.add_argument("-hM1", default=None, help="FSA-Net model 1 (headpose.models[0])")
    p.add_argument("-hM2", default=None, help="FSA-Net model 2 (headpose.models[1])")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the legacy entry point; returns the exit code (2 on input/configuration errors)."""
    args = build_parser().parse_args(None if argv is None else list(argv))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dev is not None and args.dev.replace(",", "").strip().isdigit():
        # must be set before torch initialises CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev.strip()

    from VideoProcessor import VideoProcessor, config_from_legacy_args  # noqa: PLC0415
    from vfoa.backends.base import BackendUnavailable  # noqa: PLC0415

    try:
        cfg = config_from_legacy_args(args)
        path: Optional[str] = None
        if args.dvL and args.dvR and os.path.exists(args.dvL) and os.path.exists(args.dvR):
            print("Stereo views to analyse exist")
            path = args.dvL
            cfg.input.right_path = args.dvR
            if args.dvC and os.path.isdir(args.dvC):
                cfg.input.camera_dir = args.dvC
        elif args.dvL or args.dvR:
            print(f"Stereo views not found ({args.dvL}, {args.dvR}); continuing without stereo depth")
        if path is None and args.d is not None:
            if not os.path.exists(args.d):
                print(f"File doesnt exist! ({args.d})")
                return 2
            path = args.d
        if path is None:
            if cfg.input.get("path") is None:
                print("FinalTest: error: no input (use -d VIDEO, -dvL/-dvR or --set input.path=...)",
                      file=sys.stderr)
                return 2
            path = str(cfg.input.path)
        if args.dvC and os.path.isdir(args.dvC) and not cfg.input.get("camera_dir"):
            cfg.input.camera_dir = args.dvC
        print("Process", path)
        vp = VideoProcessor(cfg)
        vp.processVideo(path, args.b, args.g, args.w, args.m)
    except (ValueError, FileNotFoundError, BackendUnavailable) as exc:
        print(f"FinalTest: error: {exc}", file=sys.stderr)
        return 2
    print(vp.report)
    print(f"results written to {vp.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
