"""Gaze proxy: use the head direction as the gaze direction with reduced confidence.

For real-time runs without a gaze network. The confidence is
``head.confidence * gaze.proxy_confidence`` so that confidence-weighted
fusion does not double count the head cue.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vfoa.backends.base import GazeBackend
from vfoa.types import DirectionEstimate, Frame, Person

__all__ = ["HeadProxyGaze"]


class HeadProxyGaze(GazeBackend):
    """Gaze = head direction (camera frame, unit) for persons with a head estimate."""

    name = "head_proxy"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        scale = float(self.section.get("proxy_confidence", 0.5))
        out: dict[int, DirectionEstimate] = {}
        for person in persons:
            head = person.head
            if head is None:
                continue
            out[person.track_id] = DirectionEstimate(
                direction=head.direction.copy(),
                confidence=float(head.confidence) * scale,
                frame_index=frame.index,
            )
        return out
