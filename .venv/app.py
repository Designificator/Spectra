from flask import Flask, render_template, Response, jsonify
import numpy as np
import cv2
from main import frame

app = Flask(__name__)

video_stream = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        ret, jpeg = cv2.imencode('.jpg', frame)

        finalframe = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + finalframe + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
     return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")