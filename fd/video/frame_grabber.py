import threading
import time
from typing import Optional
import cv2


class FrameGrabber:
    """Threaded camera reader providing the latest frame (optionally resized)."""
    def __init__(self, device_index: int, resize_width: int | None):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {device_index}")
        self.resize_width = resize_width if resize_width and resize_width > 0 else None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ret, frm = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.resize_width and frm.shape[1] > self.resize_width:
                scale = self.resize_width / float(frm.shape[1])
                frm = cv2.resize(frm, (self.resize_width, int(frm.shape[0]*scale)))
            with self.lock:
                self.frame = frm

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.cap.release()
