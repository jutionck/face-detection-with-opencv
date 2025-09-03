from typing import List
from pathlib import Path
import cv2
from .base import FaceDetector, DetectParams, Box


class DNNDetector(FaceDetector):
    def __init__(self, prototxt: str, model: str, threshold: float):
        proto = Path(prototxt)
        weights = Path(model)
        if not proto.exists() or not weights.exists():
            raise FileNotFoundError(f"Missing DNN files: {proto} / {weights}")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.threshold = threshold

    def detect(self, frame_bgr, params: DetectParams) -> List[Box]:
        (h, w) = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes: List[Box] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if bh < params.min_size_pixels:
                continue
            aspect = bw / float(bh)
            if not (params.aspect_min <= aspect <= params.aspect_max):
                continue
            boxes.append((x1, y1, bw, bh))
        return boxes
