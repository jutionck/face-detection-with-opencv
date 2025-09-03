"""Entry point for modular face detection (Haar / DNN)."""

import sys
import time
import argparse
import cv2
from fd.config import (
  WINDOW_NAME,
  DEFAULT_METHOD,
  DEFAULT_SCALE,
  DEFAULT_MIN_NEIGHBORS,
  DEFAULT_MIN_SIZE_RATIO,
  DEFAULT_ASPECT_MIN,
  DEFAULT_ASPECT_MAX,
  STABILITY_WINDOW,
  DEFAULT_DETECT_WIDTH,
  DEFAULT_DNN_MODEL,
  DEFAULT_DNN_PROTOTXT,
  DEFAULT_DNN_THRESHOLD,
)
from fd.detectors.haar import HaarDetector
from fd.detectors.dnn import DNNDetector
from fd.detectors.base import DetectParams
from fd.postprocess import Stabilizer
from fd.video.frame_grabber import FrameGrabber


def parse_args():
  p = argparse.ArgumentParser(description="Realtime Face Detection (Haar / DNN) modular")
  p.add_argument("--camera", type=int, default=0, help="Index kamera")
  p.add_argument("--method", choices=["haar", "dnn"], default=DEFAULT_METHOD, help="Metode deteksi")
  p.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Haar scaleFactor")
  p.add_argument("--neighbors", type=int, default=DEFAULT_MIN_NEIGHBORS, help="Haar minNeighbors")
  p.add_argument("--min-size-ratio", type=float, default=DEFAULT_MIN_SIZE_RATIO, help="Rasio tinggi wajah minimum (0-1)")
  p.add_argument("--aspect-min", type=float, default=DEFAULT_ASPECT_MIN, help="Rasio aspek minimum")
  p.add_argument("--aspect-max", type=float, default=DEFAULT_ASPECT_MAX, help="Rasio aspek maksimum")
  p.add_argument("--detect-width", type=int, default=DEFAULT_DETECT_WIDTH, help="Lebar resize deteksi (0=off)")
  p.add_argument("--no-thread", action="store_true", help="Matikan threaded capture")
  p.add_argument("--no-stabilize", action="store_true", help="Matikan stabilisasi bounding box")
  p.add_argument("--no-eye-check", action="store_true", help="Matikan validasi mata (Haar)")
  p.add_argument("--dnn-prototxt", default=DEFAULT_DNN_PROTOTXT, help="Path prototxt DNN")
  p.add_argument("--dnn-model", default=DEFAULT_DNN_MODEL, help="Path caffemodel DNN")
  p.add_argument("--dnn-threshold", type=float, default=DEFAULT_DNN_THRESHOLD, help="Ambang DNN (0-1)")
  return p.parse_args()


def build_detector(args):
  if args.method == "haar":
    return HaarDetector(args.scale, args.neighbors, not args.no_eye_check)
  return DNNDetector(args.dnn_prototxt, args.dnn_model, args.dnn_threshold)


def draw_overlay(frame, boxes, method_label: str, raw_count: int, fps: float, target_w, threaded: bool):
  for (x, y, w, h) in boxes:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cv2.putText(frame, f"Faces: {len(boxes)} (raw:{raw_count})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(frame, method_label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 255, 200), 1, cv2.LINE_AA)
  cv2.putText(frame, f"detW={target_w or frame.shape[1]} thr={'on' if threaded else 'off'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, f"FPS~{fps:.1f}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2, cv2.LINE_AA)


def main():
  args = parse_args()
  try:
    detector = build_detector(args)
  except Exception as e:  # noqa: BLE001
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

  stabilizer = Stabilizer(STABILITY_WINDOW) if not args.no_stabilize else None

  # Capture setup
  if args.no_thread:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
      print("[ERROR] Tidak bisa membuka kamera", file=sys.stderr)
      sys.exit(1)
    grabber = None
  else:
    grabber = FrameGrabber(args.camera, args.detect_width).start()
    cap = None

  fps_time = time.time()
  fps = 0.0

  try:
    while True:
      if grabber:
        frame_proc = grabber.read()
        if frame_proc is None:
          time.sleep(0.005)
          continue
        source_frame = frame_proc
        scale_used = 1.0
      else:
        ret, frame = cap.read()  # type: ignore
        if not ret:
          print("[WARN] Gagal baca frame. Keluar.")
          break
        source_frame = frame
        scale_used = 1.0
        if args.detect_width and frame.shape[1] > args.detect_width:
          scale_used = args.detect_width / float(frame.shape[1])
          source_frame = cv2.resize(frame, (int(frame.shape[1]*scale_used), int(frame.shape[0]*scale_used)))

      h_small = source_frame.shape[0]
      min_size_px = int(h_small * args.min_size_ratio)
      params = DetectParams(
        min_size_ratio=args.min_size_ratio,
        aspect_min=args.aspect_min,
        aspect_max=args.aspect_max,
        min_size_pixels=min_size_px,
      )
      boxes_small = detector.detect(source_frame, params)
      raw_count = len(boxes_small)
      if stabilizer:
        boxes_small = stabilizer.apply(boxes_small)

      if args.no_thread and scale_used != 1.0:
        inv = 1.0 / scale_used
        boxes = [(int(x*inv), int(y*inv), int(w*inv), int(h*inv)) for (x, y, w, h) in boxes_small]
        draw_target = frame  # original
      else:
        boxes = boxes_small
        draw_target = source_frame

      method_label = (
        f"haar s={args.scale} n={args.neighbors}" if args.method == 'haar' else f"dnn thr={args.dnn_threshold:.2f}"
      )
      draw_overlay(draw_target, boxes, method_label, raw_count, fps, args.detect_width, not args.no_thread)
      cv2.imshow(WINDOW_NAME, draw_target)

      # FPS update (half-second window)
      now = time.time()
      if now - fps_time >= 0.5:
        fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
        fps_time = now

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  finally:
    if grabber:
      grabber.stop()
    if cap:
      cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
  main()
