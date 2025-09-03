"""Simple real-time face detection using OpenCV Haar Cascade.

Perbaikan utama dibanding versi awal:
  - Menggunakan grayscale untuk deteksi (lebih akurat & ringan).
  - Validasi pemuatan model cascade dan kamera.
  - Struktur fungsi lebih modular.
  - Cleanup terjamin dengan try/finally.
  - Path model robust relatif terhadap lokasi file ini.
Tekan 'q' untuk keluar.
"""

from pathlib import Path
import sys
import cv2
import argparse
from collections import deque

# ---------------- Configuration ---------------- #
CASCADE_FILENAME = "face_ref.xml"  # Haar utama (frontal face)
WINDOW_NAME = "FaceDetection"

# Default tuning (bisa dioverride via argumen CLI)
DEFAULT_SCALE = 1.1
DEFAULT_MIN_NEIGHBORS = 6  # sedikit dinaikkan untuk kurangi false positive
DEFAULT_MIN_SIZE_RATIO = 0.08  # min tinggi wajah relatif terhadap tinggi frame
DEFAULT_ASPECT_MIN = 0.75
DEFAULT_ASPECT_MAX = 1.35
STABILITY_WINDOW = 5  # jumlah frame untuk stabilisasi (temporal smoothing)

# Optional eye cascade (second-stage validation) - gunakan yang bawaan OpenCV jika tersedia
EYE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"


def load_cascade() -> cv2.CascadeClassifier:
  """Load Haar Cascade file and validate.
  Exit program if file not found / invalid.
  """
  cascade_path = Path(__file__).parent / CASCADE_FILENAME
  if not cascade_path.exists():
    print(f"[ERROR] File model tidak ditemukan: {cascade_path}", file=sys.stderr)
    sys.exit(1)
  cascade = cv2.CascadeClassifier(str(cascade_path))
  if cascade.empty():
    print("[ERROR] Gagal memuat Haar Cascade (file korup / format salah)", file=sys.stderr)
    sys.exit(1)
  return cascade


def detect_faces(gray_frame, cascade: cv2.CascadeClassifier, *, scale, min_neighbors, min_size):
  """Return list of face bounding boxes (x,y,w,h) dengan parameter dinamis."""
  return cascade.detectMultiScale(
    gray_frame,
    scaleFactor=scale,
    minNeighbors=min_neighbors,
    flags=cv2.CASCADE_SCALE_IMAGE,
    minSize=min_size,
  )


def draw_face_boxes(frame, faces):
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def load_eye_cascade():
  if EYE_CASCADE_PATH.exists():
    eye = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
    if not eye.empty():
      return eye
  return None


def filter_candidates(candidates, frame_shape, *, aspect_min, aspect_max):
  """Filter berdasar rasio aspek agar objek non-wajah (terlalu lebar/tinggi) tereliminasi."""
  filtered = []
  for (x, y, w, h) in candidates:
    aspect = w / float(h)
    if aspect < aspect_min or aspect > aspect_max:
      continue
    # (Opsional tambahan bisa: posisi, area relative, dsb)
    filtered.append((x, y, w, h))
  return filtered


def eye_stage_validation(gray_frame, faces, eye_cascade, max_checks=6):
  """Validasi tambahan: pastikan ada minimal 1 pasang (atau 1) mata dalam region wajah.
  Membatasi jumlah region yang dicek untuk efisiensi.
  """
  if eye_cascade is None:
    return faces
  validated = []
  for idx, (x, y, w, h) in enumerate(faces):
    if idx >= max_checks:
      # Jangan cek semuanya agar tetap realtime
      validated.append((x, y, w, h))
      continue
    roi_gray = gray_frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(int(w*0.15), int(h*0.15)))
    if len(eyes) >= 1:  # minimal terdeteksi 1 mata
      validated.append((x, y, w, h))
  return validated


def stabilize_detections(history_deque, current_faces):
  """Sederhana: hanya gunakan face yang konsisten muncul beberapa frame.
  Menghitung frekuensi kasar bounding box (aproksimasi posisi) dalam window.
  """
  history_deque.append(current_faces)
  # Flatten
  all_faces = [f for frame_faces in history_deque for f in frame_faces]
  stabilized = []
  for (x, y, w, h) in current_faces:
    # Hitung berapa kali kemunculan (dengan toleransi posisi)
    count = 0
    for (x2, y2, w2, h2) in all_faces:
      if abs(x - x2) < w * 0.25 and abs(y - y2) < h * 0.25:
        count += 1
    if count >= max(2, len(history_deque)//2):  # ambang dinamis
      stabilized.append((x, y, w, h))
  return stabilized


def open_camera(device_index: int = 0) -> cv2.VideoCapture:
  cap = cv2.VideoCapture(device_index)
  if not cap.isOpened():
    print(f"[ERROR] Kamera dengan index {device_index} tidak dapat dibuka", file=sys.stderr)
    sys.exit(1)
  return cap


def parse_args():
  p = argparse.ArgumentParser(description="Realtime Face Detection (Haar) dengan filter false positive")
  p.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
  p.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="scaleFactor untuk Haar (rendah=lebih akurat tapi lambat)")
  p.add_argument("--neighbors", type=int, default=DEFAULT_MIN_NEIGHBORS, help="minNeighbors (lebih tinggi=lebih sedikit false positive)")
  p.add_argument("--min-size-ratio", type=float, default=DEFAULT_MIN_SIZE_RATIO, help="Rasio minimum tinggi wajah terhadap tinggi frame (0-1)")
  p.add_argument("--aspect-min", type=float, default=DEFAULT_ASPECT_MIN, help="Rasio aspek minimum (w/h)")
  p.add_argument("--aspect-max", type=float, default=DEFAULT_ASPECT_MAX, help="Rasio aspek maksimum (w/h)")
  p.add_argument("--no-eye-check", action="store_true", help="Matikan validasi mata (lebih cepat, mungkin lebih banyak false positive)")
  p.add_argument("--no-stabilize", action="store_true", help="Matikan stabilisasi temporal")
  return p.parse_args()


def main(device_index: int = 0):  # device_index dipakai bila langsung dipanggil tanpa CLI
  args = parse_args() if len(sys.argv) > 1 else argparse.Namespace(
    camera=device_index,
    scale=DEFAULT_SCALE,
    neighbors=DEFAULT_MIN_NEIGHBORS,
    min_size_ratio=DEFAULT_MIN_SIZE_RATIO,
    aspect_min=DEFAULT_ASPECT_MIN,
    aspect_max=DEFAULT_ASPECT_MAX,
    no_eye_check=False,
    no_stabilize=False,
  )

  cascade = load_cascade()
  eye_cascade = None if args.no_eye_check else load_eye_cascade()
  cap = open_camera(args.camera)
  history = deque(maxlen=STABILITY_WINDOW)

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        print("[WARN] Gagal membaca frame dari kamera. Mengakhiri.")
        break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      h_frame, w_frame = gray.shape[:2]
      min_h = int(h_frame * args.min_size_ratio)
      min_size = (min_h, min_h)

      raw_faces = detect_faces(gray, cascade, scale=args.scale, min_neighbors=args.neighbors, min_size=min_size)
      filtered = filter_candidates(raw_faces, gray.shape, aspect_min=args.aspect_min, aspect_max=args.aspect_max)
      validated = eye_stage_validation(gray, filtered, eye_cascade)
      if not args.no_stabilize:
        final_faces = stabilize_detections(history, validated)
      else:
        final_faces = validated

      draw_face_boxes(frame, final_faces)

      cv2.putText(frame, f"Faces: {len(final_faces)} (raw:{len(raw_faces)})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(frame, f"scale={args.scale} neigh={args.neighbors}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

      cv2.imshow(WINDOW_NAME, frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  finally:
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover (manual run)
  main()
