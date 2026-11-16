"""CPU thread settings of the fast backends (fakes only; no real models, no durations).

* ``_ultra.load_yolo`` resets torch's intra-op thread count after the
  warm-up predict, because ultralytics raises it at the first predict.
* With ``segmentation.async`` the CPU default is a quarter (not half) of the
  logical CPUs, because the segmentation worker runs a second torch pool.
* The l2cs ONNX Runtime session disables intra-op spin-waiting.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest

from vfoa.backends.fast import _ultra
from vfoa.backends.fast import l2cs_gaze as L


class FakeTorch:
    def __init__(self, threads: int = 7) -> None:
        self.threads = threads
        self.calls: list[int] = []

    def get_num_threads(self) -> int:
        return self.threads

    def set_num_threads(self, n: int) -> None:
        self.calls.append(n)
        self.threads = n


def _fake_yolo(events: list, torch: FakeTorch | None):
    class FakeYOLO:
        def __init__(self, path: str) -> None:
            events.append(("init", path))

        def predict(self, image, verbose=False, **kwargs):
            # ultralytics' select_device() raises the thread count at the first predict
            if torch is not None:
                torch.threads = 7
            events.append(("predict", kwargs.get("device"), list(torch.calls) if torch else None))
            return []

    return FakeYOLO


def _load(make_config, tmp_path, monkeypatch, torch, num_threads=None, device="cpu", seg_async=False):
    weights = tmp_path / "yolo11n-pose.pt"
    weights.write_bytes(b"x")
    events: list = []
    monkeypatch.setattr(_ultra, "import_yolo", lambda: _fake_yolo(events, torch))
    if torch is None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "torch", torch)
    cfg = make_config("fast", {"runtime": {"num_threads": num_threads}, "segmentation": {"async": seg_async}})
    _ultra.load_yolo(cfg, str(weights), "persons.model", imgsz=64, device=device)
    return events


def test_num_threads_set_after_warmup(make_config, tmp_path, monkeypatch):
    torch = FakeTorch()
    events = _load(make_config, tmp_path, monkeypatch, torch, num_threads=3)
    assert torch.calls == [3]
    # the warm-up predict ran before set_num_threads
    assert [e[0] for e in events] == ["init", "predict"] and events[1][2] == []
    assert torch.threads == 3


def test_cpu_default_caps_at_half_the_cpus(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    torch = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch, num_threads=None, device="cpu")
    assert torch.calls == [4]
    torch2 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch2, num_threads=None, device=None)
    assert torch2.calls == [4]
    # never raises the count above what torch already uses
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    torch3 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch3, num_threads=None, device="cpu:0")
    assert torch3.calls == [7]
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    torch4 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch4, num_threads=None, device="cpu")
    assert torch4.calls == [1]


def test_gpu_device_leaves_threads_alone(make_config, tmp_path, monkeypatch):
    torch = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch, num_threads=None, device="cuda")
    assert torch.calls == []
    torch2 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch2, num_threads=2, device="cuda:0")
    assert torch2.calls == [2]  # an explicit setting always applies


def test_no_torch_does_nothing(make_config, tmp_path, monkeypatch):
    events = _load(make_config, tmp_path, monkeypatch, None, num_threads=3)
    assert [e[0] for e in events] == ["init", "predict"]
    assert "torch" not in sys.modules
    assert _ultra.apply_torch_threads({"runtime": {"num_threads": 3}}, "cpu") is None


# ------------------------------------------------------------------------- async segmentation
def test_async_segmentation_caps_at_a_quarter_of_the_cpus(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    for device in ("cpu", None):
        torch = FakeTorch()
        _load(make_config, tmp_path, monkeypatch, torch, device=device, seg_async=True)
        assert torch.calls == [2], device
        torch2 = FakeTorch()
        _load(make_config, tmp_path, monkeypatch, torch2, device=device, seg_async=False)
        assert torch2.calls == [4], device


def test_async_cap_on_small_machines(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    torch = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch, seg_async=True)
    assert torch.calls == [1]
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    torch2 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch2, seg_async=True)
    assert torch2.calls == [1]


def test_async_cap_never_raises_the_count(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    torch = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch, seg_async=True)
    assert torch.calls == [7]  # 64 // 4 = 16, but the warm-up left 7
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    torch2 = FakeTorch(threads=1)
    monkeypatch.setitem(sys.modules, "torch", torch2)
    cfg = make_config("fast", {"segmentation": {"async": True}})
    assert _ultra.apply_torch_threads(cfg, "cpu") == 1
    assert torch2.calls == [1]


def test_explicit_num_threads_wins_with_async(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    for seg_async in (True, False):
        torch = FakeTorch()
        _load(make_config, tmp_path, monkeypatch, torch, num_threads=3, seg_async=seg_async)
        assert torch.calls == [3], seg_async
    torch2 = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch2, num_threads=3, device="cuda", seg_async=True)
    assert torch2.calls == [3]


def test_gpu_with_async_leaves_threads_alone(make_config, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    torch = FakeTorch()
    _load(make_config, tmp_path, monkeypatch, torch, device="cuda", seg_async=True)
    assert torch.calls == []


def test_no_torch_with_async(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert _ultra.apply_torch_threads({"segmentation": {"async": True}}, "cpu") is None
    assert "torch" not in sys.modules


def test_plain_mapping_and_log_reason(monkeypatch, caplog):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(threads=8))
    caplog.set_level(logging.INFO, logger="vfoa.backends.fast._ultra")
    assert _ultra.apply_torch_threads({"segmentation": {"async": True}}, "cpu") == 2
    assert "segmentation.async" in caplog.records[-1].getMessage()
    caplog.clear()
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(threads=8))
    assert _ultra.apply_torch_threads({}, "cpu") == 4
    message = caplog.records[-1].getMessage()
    assert "set to 4" in message and "segmentation.async" not in message


# ------------------------------------------------------------------------- l2cs / onnxruntime
class _Session:
    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 3, 448, 448])]

    def get_outputs(self):
        return [SimpleNamespace(name="yaw"), SimpleNamespace(name="pitch")]


def _l2cs_cfg(make_config, tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir(exist_ok=True)
    (wdir / "mobileone_s0_gaze.onnx").write_bytes(b"x")
    return make_config("fast", {"models": {"weights_dir": str(wdir), "allow_download": False, "device": "cpu"}})


def test_l2cs_disables_ort_spinning(make_config, tmp_path, monkeypatch):
    seen = {}

    class Options:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}
            self.intra_op_num_threads = 0

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.entries[key] = value

    class FakeOrt:
        SessionOptions = Options

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            seen["options"] = sess_options
            return _Session()

    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr(L, "_import_ort", lambda: FakeOrt)
    be = L.BinnedGaze360Backend(_l2cs_cfg(make_config, tmp_path))
    be.load()
    assert seen["options"].entries == {"session.intra_op.allow_spinning": "0"}


def test_l2cs_loads_without_config_entries(make_config, tmp_path, monkeypatch):
    class FakeOrt:
        SessionOptions = SimpleNamespace  # no add_session_config_entry

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            return _Session()

    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr(L, "_import_ort", lambda: FakeOrt)
    be = L.BinnedGaze360Backend(_l2cs_cfg(make_config, tmp_path))
    be.load()
    assert be.is_loaded


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
