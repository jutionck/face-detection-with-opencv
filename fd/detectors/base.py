from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # x, y, w, h

@dataclass
class DetectParams:
    min_size_ratio: float
    aspect_min: float
    aspect_max: float
    min_size_pixels: int

class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr, params: DetectParams) -> List[Box]:
        ...
