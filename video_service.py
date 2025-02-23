from flask import Flask, request, jsonify
import os
import gdown
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import Counter

app = Flask(__name__)

FER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_model.h5")
emotion_model = load_model(FER_MODEL_PATH)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Video Analysis API is running!"})

@app.route("/analyze-video", methods=["POST"])
def analyze_video_route():
    data = request.json
    video_link = data.get("video_link")

    if not video_link:
        return jsonify({"error": "Missing video link"}), 400

    video_path = download_video(video_link)
    video_analysis = analyze_video(video_path)

    return jsonify(video_analysis)

def download_video(url):
    output_path = "temp_video.mp4"
    os.system(f"gdown {url} -O {output_path}")
    return output_path

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0]), \
                             int(bboxC.width * frame.shape[1]), int(bboxC.height * frame.shape[0])
                face_crop = frame[y:y + h, x:x + w]

                face_crop = cv2.resize(face_crop, (48, 48))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = face_crop / 255.0
                face_crop = np.reshape(face_crop, (1, 48, 48, 1))

                prediction = emotion_model.predict(face_crop)
                emotion_label = EMOTION_LABELS[np.argmax(prediction)]
                emotions.append(emotion_label)

    cap.release()
    return {"most_frequent_emotion": Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"}

def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(debug=True)
