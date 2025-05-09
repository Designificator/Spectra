
from encodings.utf_7 import encode
from idlelib.iomenu import encoding
import json

from flask import Flask, render_template, Response, jsonify
import numpy as np
import cv2
from main import freshest as cam, draw_tracks, update_visitors, process_image
from itertools import zip_longest

from matplotlib.font_manager import json_dump

app = Flask(__name__)
tracks = {}
SUM = [0]
#cam.set(cv2.CAP_PROP_BUFFERSIZE, 4)
@app.route('/')
def index():
    return render_template('html.html')

def gencam(sum, camera, tracks):
    while True:

        ret, frame = camera.read()
        if not ret:
            break
        frame, tracks = process_image(frame, tracks)
        frame = draw_tracks(frame, tracks)
        frame = cv2.line(frame, (500, 0), (500, 1000), (0, 0, 255), 3)

        sum[0] = update_visitors(sum[0], tracks)
        ret, jpeg = cv2.imencode('.jpg', frame)
        finalframe = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + finalframe + b'\r\n\r\n')
def gencount(sum, tracks):
    while True:
        sum = update_visitors(sum, tracks)
        sum_b = json.dumps({"count" : sum[0]}).encode()
        yield (sum_b)

@app.route('/video_feed')
def video_feed():
    return Response(gencam(SUM, cam, tracks),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/count')
def count():
    return Response(next(gencount(SUM, tracks)), mimetype='application/json')
    return Response(json.dumps({"count": random.randint(1, 99)}).encode())
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")