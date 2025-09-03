from typing import List
from pathlib import Path
import cv2
from .base import FaceDetector, DetectParams, Box
from ..config import CASCADE_FILENAME, EYE_CASCADE_PATH, project_path


class HaarDetector(FaceDetector):
    def __init__(self, scale_factor: float, min_neighbors: int, eye_check: bool):
        cascade_path = project_path(CASCADE_FILENAME)
        if not cascade_path.exists():
            raise FileNotFoundError(f"Cascade not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError("Failed to load cascade")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.eye_check = eye_check
        self.eye_cascade = None
        if eye_check and EYE_CASCADE_PATH.exists():
            eye = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
            if not eye.empty():
                self.eye_cascade = eye

    def _eye_filter(self, gray, boxes: List[Box]) -> List[Box]:
        if not self.eye_cascade:
            return boxes
        approved: List[Box] = []
        for (x, y, w, h) in boxes:
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(int(w*0.15), int(h*0.15)))
            if len(eyes) >= 1:
                approved.append((x, y, w, h))
        return approved

    def detect(self, frame_bgr, params: DetectParams) -> List[Box]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        min_size = (params.min_size_pixels, params.min_size_pixels)
        raw = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size,
        )
        out: List[Box] = []
        for (x, y, w, h) in raw:
            if h < params.min_size_pixels:
                continue
            aspect = w / float(h)
            if not (params.aspect_min <= aspect <= params.aspect_max):
                continue
            out.append((x, y, w, h))
        return self._eye_filter(gray, out)
