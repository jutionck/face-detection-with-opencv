from collections import deque
from typing import List
from .detectors.base import Box


class Stabilizer:
    def __init__(self, window: int):
        self.history = deque(maxlen=window)

    def apply(self, boxes: List[Box]) -> List[Box]:
        self.history.append(boxes)
        flattened = [b for frame_boxes in self.history for b in frame_boxes]
        stable: List[Box] = []
        for (x, y, w, h) in boxes:
            count = 0
            for (x2, y2, w2, h2) in flattened:
                if abs(x - x2) < w * 0.25 and abs(y - y2) < h * 0.25:
                    count += 1
            if count >= max(2, len(self.history)//2):
                stable.append((x, y, w, h))
        return stable
