import cv2
import supervision as sv
from inference import InferencePipeline
from stream import update_frame
from esp32_utils import fetch_sensor_data
import os
import threading
from datetime import datetime, timedelta

# --- Configuration ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cooldown_seconds = 10  # seconds between allowed incident reports

# --- Annotators ---
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

# --- Globals ---
incident_logs = []
last_incident_time = None

# -----------------------------
# Helper: Async image + sensor saving
# -----------------------------
def save_incident_async(image):
    """Save incident image and fetch sensor data in background thread."""
    def worker():
        try:
            os.makedirs("static/pending_incidents", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_vehicle_incident.jpg"
            save_path = os.path.join("static/pending_incidents", filename)

            # Save annotated frame
            cv2.imwrite(save_path, image)
            print(f"[✅ Saved] Incident image saved at: {save_path}")

            # Fetch sensor data safely
            sensor_data = fetch_sensor_data()
            if sensor_data:
                print("[⚠️ Incident Detected] Sensor Data:", sensor_data)
        except Exception as e:
            print("[❌ Error saving incident]:", e)

    threading.Thread(target=worker, daemon=True).start()


# -----------------------------
# Main callback from inference
# -----------------------------
def my_custom_sink(predictions: dict, video_frame):
    global last_incident_time

    # Extract detections + labels
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)

    # Resize and annotate frame
    frame = video_frame.image

    # Ensure proper format for RTSP sources
    if frame.ndim == 2:  # grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.dtype != "uint8":
        frame = cv2.convertScaleAbs(frame)  # ensure 8-bit depth

    resized_image = cv2.resize(frame, (640, 480))#for webcam, simplicity

    print("First detection box:", detections.xyxy[0] if len(detections.xyxy) > 0 else "None")#For debugging purposes

    # Draw boxes first, then labels (better visibility)                                           # ---------------------#
    annotated = box_annotator.annotate(scene=resized_image, detections=detections)               # WEBCAM CONFIGURATION #
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)  # -------------------- #
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    annotated_small = cv2.resize(annotated, (640, 480))
    update_frame(annotated_small)

    # Update stream display
    update_frame(annotated)

    # Incident detection logic
    if "vehicle_incident" in labels:
        now = datetime.now()
        if last_incident_time and (now - last_incident_time).total_seconds() < cooldown_seconds:
            # Within cooldown window, ignore
            return

        print("[⚠️ Incident Detected] Vehicle incident detected (cooldown active).")
        incident_logs.append(">> Vehicle incident detected! Please verify!")

        # Run heavy tasks asynchronously
        save_incident_async(annotated.copy())

        # Update cooldown *after* triggering async save
        last_incident_time = now


# -----------------------------
# Retrieve and clear logs
# -----------------------------
def get_incident_logs():
    global incident_logs
    logs = incident_logs[:]
    incident_logs = []
    return logs


# -----------------------------
# Start the inference pipeline
# -----------------------------
def start_pipeline():
    pipeline = InferencePipeline.init(
        model_id="traffic-accident-detection-xyood/1",
        video_reference="rtsp://Accitrack;admin123@192.168.100.46/stream1",
        #video_reference=0,  # Use webcam instead if needed
        on_prediction=my_custom_sink
    )
    pipeline.start()
