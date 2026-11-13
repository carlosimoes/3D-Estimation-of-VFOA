"""Legacy thesis inter-frame tracker: thin wrapper over :class:`vfoa.tracking.PersonTracker`.

``Tracker`` and ``TrackerMatcher`` keep their thesis API but the tracker state
(and the id counter) persists across ``trackInterFrames`` calls, so ids never
collide between batches (issues #12, #18). Frame keys of the returned
:class:`Data.MapFramesOutputs` are local to the batch (0..n-1); ids are global.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

import logging
from time import perf_counter

import numpy as np

from Data import MapFramesOutputs
from vfoa.tracking import PersonTracker, box_iou
from vfoa.types import PersonDetection

logger = logging.getLogger(__name__)

#: OpenPose (CMU 18-part) head parts: REye, REar, LEye, LEar, Nose.
HEAD_PARTS = [14, 16, 15, 17, 0]


class Tracker:
    """IoU head-box tracker (thesis behaviour: previous update only, IoU > 0.5)."""

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.currentState: dict = {}
        self._tracker = PersonTracker(type="greedy_iou", iou_thr=0.5, max_age=0)
        self._frame = 0

    def setDim(self, w, h) -> None:
        self.width, self.height = w, h

    def reset(self) -> None:
        """Start a new stream: ids restart at 0."""
        self._tracker.reset()
        self._frame = 0
        self.currentState = {}

    def _track(self, entries: list) -> dict:
        """Track one frame of ``(detection, payload)`` pairs -> {id: payload}."""
        dets = [d for d, _ in entries]
        by_det = {id(d): payload for d, payload in entries}
        persons = self._tracker.update(dets, self._frame)
        self._frame += 1
        out = {}
        for person in persons:
            out[person.track_id] = by_det[id(person.detection)]
        return out

    def trackInterFrames(self, listFramesKeypoints, listFramesBoxes) -> MapFramesOutputs:
        t = perf_counter()
        result = MapFramesOutputs()
        n = 0
        for i, listBoxes in enumerate(listFramesBoxes):
            entries = [(PersonDetection(head_box=box), {"b": box})
                       for box in (listBoxes or []) if box is not None]
            current = self._track(entries)
            self.currentState = current
            result.map[i] = current
            n += 1
        t = perf_counter() - t
        logger.info("Interframe tracking: %.4f s; %.1f frames per second", t, n / t if t > 0 else 0.0)
        return result

    def find_id(self, currentBox, boxes):
        """Id of the box in ``boxes`` ({id: {"b": box}}) with the best IoU > 0.5, else None."""
        id_final = None
        max_iou = 0.5
        for k, box in boxes.items():
            iou = self.intersectionOverUnionRatio(currentBox, box["b"])
            if iou > max_iou:
                id_final, max_iou = k, iou
        return id_final

    def intersectionOverUnionRatio(self, bb1, bb2) -> float:
        return box_iou(bb1, bb2)


class TrackerMatcher(Tracker):
    """Matches OpenPose skeletons to head boxes, then tracks the boxes."""

    def __init__(self) -> None:
        super().__init__()

    def _head_points(self, human) -> np.ndarray:
        sx = self.width if self.width else 1.0
        sy = self.height if self.height else 1.0
        pts = [(part.x * sx, part.y * sy) for i, part in getattr(human, "body_parts", {}).items()
               if i in HEAD_PARTS]
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def matchKeypointsBoxes(self, listPoints, listBoxes) -> dict:
        """{human: head box | None}; a box contains ALL head keypoints and is claimed once."""
        humans = list(listPoints or [])
        boxes = list(listBoxes or [])
        result = {human: None for human in humans}
        pairs = []
        for hi, human in enumerate(humans):
            pts = self._head_points(human)
            if pts.shape[0] == 0:
                continue
            centre = pts.mean(axis=0)
            for bi, box in enumerate(boxes):
                if box is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                inside = np.all((pts[:, 0] >= x1) & (pts[:, 0] <= x2)
                                & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
                if inside:
                    dist = float(np.hypot(centre[0] - (x1 + x2) / 2, centre[1] - (y1 + y2) / 2))
                    pairs.append((dist, hi, bi))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        used_h, used_b = set(), set()
        for _, hi, bi in pairs:
            if hi in used_h or bi in used_b:
                continue
            used_h.add(hi)
            used_b.add(bi)
            result[humans[hi]] = boxes[bi]
        return result

    def trackInterFrames(self, listFramesKeypoints, listFramesBoxes) -> MapFramesOutputs:
        t = perf_counter()
        tracking_id = MapFramesOutputs()
        n = 0
        for i, (listKeypoints, listBoxes) in enumerate(zip(listFramesKeypoints, listFramesBoxes)):
            matched = self.matchKeypointsBoxes(listKeypoints, listBoxes)
            entries = []
            for human, box in matched.items():
                if box is not None:
                    entries.append((PersonDetection(head_box=box), {"kp": human, "b": box}))
                else:
                    entries.append((PersonDetection(head_box=None), {"kp": human}))
            current = self._track(entries)
            self.currentState = {k: v["b"] for k, v in current.items() if "b" in v}
            tracking_id.addListEntries(i, current)
            n += 1
        t = perf_counter() - t
        logger.info("Interframe tracking: %.4f s; %.1f frames per second", t, n / t if t > 0 else 0.0)
        return tracking_id

    def findId(self, currentBox):
        id_final = None
        max_iou = 0.5
        for k, box in self.currentState.items():
            iou = self.intersectionOverUnionRatio(currentBox, box)
            if iou > max_iou:
                id_final, max_iou = k, iou
        return id_final
