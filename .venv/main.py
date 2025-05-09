from operator import truediv

import cv2, queue, threading, time

from ultralytics import YOLO
import cv2
import numpy as np
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    config = load_dotenv(dotenv_path)
# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')
cam_ip = os.environ.get("CAM_IP")
cam_port = os.environ.get("CAM_PORT")
cam_attr = os.environ.get("CAM_CAPTURE_ATTR")
#for i in range(1,300):
capture_url = 'rtsp://admin:admin@'+cam_ip+':'+cam_port+"/"+cam_attr+".sdp"


class Camera:
    last_frame = None
    last_ready = None
    last_ready2 = None
    lock = Lock()
    capture = None

    def __init__(self, link):
        self.capture = cv2.VideoCapture(link)
        self.signal = 1
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready = self.capture.grab()

    def getFrame(self):
        if (self.last_ready):
            self.last_ready2, self.last_frame = self.capture.retrieve()
            if (self.last_ready2):
                return self.last_frame.copy()
            else:
                return -1
        else:
            return -1

    def delete(self):
        self.capture.release()

cam = cv2.VideoCapture(capture_url) #capture_url

# Функция для обработки изображения
def process_image(image, tracks):
    # Загрузка изображения
    results = model.track(image, persist=True, verbose=False)

    # Получение оригинального изображения и результатов
    if results[0].boxes.id == None: return image, tracks

    image = results[0].orig_img
    classes_names = results[0].names
    classes = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    boxes_id = results[0].boxes.id.int().cpu().tolist()

    people = []
    # Рисование рамок и группировка результатов
    for class_id, box, box_id in zip(classes, boxes, boxes_id):
        class_name = classes_names[int(class_id)]
        color = (255, 255, 255)  # Выбор цвета для класса
        if class_name != "person":
            continue
        people.append((box, box_id))

        # Рисование рамок на изображении
        x1, y1, x2, y2 = box
        centre = (int((x1 + x2)/2), int((y1 + y2)/2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name + str(box_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        track = tracks.get(box_id)
        if track == None: track = []
        elif len(track) == 30: track.pop(0)
        track.append(centre)
        tracks[box_id] = track
    tracks = delete_old_tracks(tracks, boxes_id)
    return image, tracks

def draw_tracks(image, tracks):
    for key, track in tracks.items():
        for point_index in range(1, len(track)):
            pt1 = track[point_index - 1]
            pt2 = track[point_index]
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    return image
def delete_old_tracks(tracks, keys):
    tracks = {key: tracks[key] for key in keys if key in tracks}
    return tracks

def border_line_func(x, y):
    if x >= 500: return True
    else: return False

def count_if_crossed(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    if border_line_func(x1, y1) and not border_line_func(x2, y2):
        return -1
    elif border_line_func(x2, y2) and not border_line_func(x1, y2):
        return 1
    else: return 0

def update_visitors(sum, tracks):
    for key, track in tracks.items():
        if len(track) < 2: continue
        pt1 = track[-2]
        pt2 = track[-1]
        sum += count_if_crossed(pt1, pt2)
    if sum < 0:
        print("ERROR, negative sum!")
        sum = 0
    print(sum)
    return sum

tracks = {}
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
#     small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#
#     frame, tracks = process_image(frame, tracks)
#     frame = draw_tracks(frame, tracks)
#     frame = cv2.line(frame, (500, 0), (500, 1000), (0, 0, 255), 3)
#     SUM = update_visitors(SUM, tracks)
#     print(SUM)
#     cv2.imshow("video", frame)
#     if cv2.waitKey(1) == ord("q"): break
# cam.release()
# cv2.destroyAllWindows()