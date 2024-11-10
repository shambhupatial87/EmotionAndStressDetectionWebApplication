from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import dlib
import time
from emotion_classification import detect_emotion
from stress_detection import detect_stress

app = Flask(__name__)

def gen_emotion():
    cap = cv2.VideoCapture(0)
    while True: 
        ret, frame = cap.read() 
        if not ret:
            break

        labels = detect_emotion(frame)

        for label, (x, y) in labels:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_stress():
    cap = cv2.VideoCapture(0)
    data_buffer = [] 
    times = [] 
    t0 = time.time() 
    while True: 
        ret, frame = cap.read() 
        if not ret:
            break

        frame = detect_stress(frame, data_buffer, times, t0)  # Process the frame with stress detection

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion_classification')
def emotion_classification():
    return render_template('emotion_classification.html')

@app.route('/stress_detection')
def stress_detection():
    return render_template('stress_detection.html')


@app.route('/video_feed_emotion')
def video_feed_emotion():
    return Response(gen_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_stress')
def video_feed_stress():
    return Response(gen_stress(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
