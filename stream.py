import cv2
import time

# Shared frame buffer
latest_frame = None

def update_frame(frame):
    global latest_frame
    latest_frame = frame

def generate_stream():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.016)  # ~60 FPS