from pathlib import Path
import cv2

# General
WINDOW_NAME = "FaceDetection"
CASCADE_FILENAME = "face_ref.xml"

# Haar defaults
DEFAULT_SCALE = 1.1
DEFAULT_MIN_NEIGHBORS = 6
DEFAULT_MIN_SIZE_RATIO = 0.08
DEFAULT_ASPECT_MIN = 0.75
DEFAULT_ASPECT_MAX = 1.35

# Stabilization / resizing
STABILITY_WINDOW = 5
DEFAULT_DETECT_WIDTH = 640

# Method selection
DEFAULT_METHOD = "haar"  # or "dnn"

# DNN defaults (user must supply files)
DEFAULT_DNN_PROTOTXT = "deploy.prototxt"
DEFAULT_DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DEFAULT_DNN_THRESHOLD = 0.55

EYE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"

def project_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / name
