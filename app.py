"""
app.py — LensBeat Flask/SocketIO web layer

Deliberately thin: all measurement logic lives in inference/pipeline.py and
the modules it composes. Supports both:
  1. Client-Side Streaming (Selfie / Mobile Camera) via WebSocket frames
  2. Server-Side Local Capture (PC Webcam) via OpenCV VideoCapture
"""

import base64
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

from inference.pipeline import LensBeatPipeline

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "LensBeat_PhysNet_finetuned.pth"
FRAME_SIZE = 72
CHUNK_LENGTH = 128
FPS = 30
BPM_LOW_HZ = 0.83
BPM_HIGH_HZ = 2.0
FILTER_LOW_HZ = 0.7
FILTER_HIGH_HZ = 3.0
CAMERA_INDEX = 0
INFER_STRIDE = 32
ROI_MODE = "center_crop"
# =============================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "lensbeat-fusion"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

if not os.path.exists(MODEL_PATH):
    print(f"WARNING: {MODEL_PATH} not found. Copy your fine-tuned .pth here before starting.")

pipeline = LensBeatPipeline(
    model_path=MODEL_PATH,
    frame_size=FRAME_SIZE,
    chunk_length=CHUNK_LENGTH,
    fs=FPS,
    bpm_low_hz=BPM_LOW_HZ,
    bpm_high_hz=BPM_HIGH_HZ,
    filter_low_hz=FILTER_LOW_HZ,
    filter_high_hz=FILTER_HIGH_HZ,
    roi_mode=ROI_MODE,
    infer_stride=INFER_STRIDE,
)
print(f"Pipeline ready on {pipeline.device}, ROI mode = {ROI_MODE}")


class WebcamWorker:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _loop(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        if not cap.isOpened():
            socketio.emit("error", {"message": "Could not open local server webcam"})
            self.running = False
            return

        socketio.emit("status", {"message": "Server webcam started"})

        while self.running:
            ret, frame = cap.read()
            if not ret:
                socketio.emit("error", {"message": "Lost webcam frame"})
                break

            preview_frame = cv2.resize(frame, (400, 300))
            _, buf = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_img = base64.b64encode(buf).decode('utf-8')
            socketio.emit("video_frame", {"image": f"data:image/jpeg;base64,{b64_img}"})

            try:
                result = pipeline.process_frame(frame)
                if result is not None:
                    result["timestamp"] = time.time()
                    socketio.emit("bpm_update", result)
            except Exception as e:
                socketio.emit("error", {"message": f"Inference error: {e}"})

            socketio.sleep(1.0 / FPS)

        cap.release()
        socketio.emit("status", {"message": "Webcam stopped"})


worker = WebcamWorker()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("start_monitoring")
def handle_start():
    worker.start()


@socketio.on("stop_monitoring")
def handle_stop():
    worker.stop()


@socketio.on("client_frame")
def handle_client_frame(data):
    """Processes incoming video frames streamed from client browser/mobile selfie camera."""
    try:
        raw_b64 = data.get("image", "")
        if not raw_b64:
            return
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            result = pipeline.process_frame(frame)
            if result is not None:
                result["timestamp"] = time.time()
                socketio.emit("bpm_update", result)
    except Exception as e:
        socketio.emit("error", {"message": f"Client frame error: {e}"})


if __name__ == "__main__":
    ssl_ctx = None
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_ctx = ("cert.pem", "key.pem")
        print("Starting HTTPS server with SSL certificates (for mobile camera access)...")
    else:
        print("Starting HTTP server (Warning: mobile cameras require HTTPS to enable getUserMedia)...")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_ctx, allow_unsafe_werkzeug=True)
