import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import threading

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "car_accident_model.weights.h5"
DEFAULT_VIDEO_PATH = BASE_DIR / "Accident_Detection.mp4"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def build_model():
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='Same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

class VideoDetector:
    def __init__(self):
        print("[INFO] Loading Model...")
        self.model = build_model()
        self.model.load_weights(str(WEIGHTS_PATH))
        self.source = str(DEFAULT_VIDEO_PATH)
        self.cap = None
        self.lock = threading.Lock()
        self.output_frame = None
        self.threshold = 0.85
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=30, detectShadows=False)
        self._init_capture()

    def _init_capture(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=30, detectShadows=False)

    def set_source(self, source_path):
        with self.lock:
            self.source = str(source_path)
            self._init_capture()

    def run(self):
        target_delay = 1.0 / 30.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        while True:
            loop_start = time.time()
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # Loop video when it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (800, 480))
            h, w = frame.shape[:2]

            # 1. Model Prediction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            tensor = (resized.astype("float32") / 255.0).reshape(1, 128, 128, 1)

            pred = self.model(tensor, training=False).numpy()[0]
            acc_prob = float(pred[0])  # Class 0: Accident
            is_accident = (acc_prob >= self.threshold)

            # Colors: Green when safe, Red when accident
            # OpenCV uses (Blue, Green, Red)
            if is_accident:
                color = (0, 0, 255)  # Red
                conf = acc_prob * 100.0
                text = f"Accident: {conf:.1f}%"
            else:
                color = (0, 255, 0)  # Green
                conf = (1.0 - acc_prob) * 100.0
                text = f"Safe: {conf:.1f}%"

            # 2. Bounding boxes on moving vehicles/objects in video
            fg = self.bg_subtractor.apply(frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if 500 < area < 70000:
                    x, y, bw, bh = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # 3. Side edges become Red when accident and Green when safe
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)

            # 4. Display confidence directly inside the video
            cv2.rectangle(frame, (10, 10), (280, 52), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            with self.lock:
                self.output_frame = frame.copy()

            elapsed = time.time() - loop_start
            wait = target_delay - elapsed
            if wait > 0.005:
                time.sleep(wait)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
detector = VideoDetector()

def generate_stream():
    while True:
        with detector.lock:
            if detector.output_frame is None:
                time.sleep(0.02)
                continue
            ret, jpeg = cv2.imencode(".jpg", detector.output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" in request.files:
        file = request.files["video"]
        if file and file.filename != "":
            save_path = UPLOADS_DIR / f"upload_{int(time.time())}_{file.filename}"
            file.save(str(save_path))
            detector.set_source(save_path)
    return redirect("/")

@app.route("/reset", methods=["POST"])
def reset_video():
    detector.set_source(DEFAULT_VIDEO_PATH)
    return redirect("/")

import socket
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port number (default 8000)")
    args = parser.parse_args()

    port = args.port
    # Automatically find an open port if specified port is busy
    for p in range(port, port + 10):
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(("0.0.0.0", p))
            test_socket.close()
            port = p
            break
        except OSError:
            continue

    t = threading.Thread(target=detector.run, daemon=True)
    t.start()
    print(f"\n[INFO] Starting server at: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
