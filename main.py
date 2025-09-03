import cv2

# Referensi pengenalan wajah
face_ref = cv2.CascadeClassifier('face_ref.xml')

# Deteksi kamera
camera = cv2.VideoCapture(0) # deteksi default untuk kamera bawaan laptop

def face_detection(frame):
    # Tips frame dibuat black and white untuk deteksi wajah agar ringan
  optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5)
  return faces

def drawer_box(frame):
  for x, y, w, h in face_detection(frame):
    # (0, 0, 255) -> blue, green, red
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
  pass

def close_window():
  camera.release()
  cv2.destroyAllWindows()
  exit()

def main():
  while True:
    _, frame = camera.read()
    drawer_box(frame)
    cv2.imshow("FaceDetection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      close_window()

if __name__ == "__main__":
  main()
