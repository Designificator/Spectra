import cv2

from ultralytics import YOLO
import cv2
import numpy as np
import os

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')
cam_ip = os.environ.get("CAM_IP")
cam_port = os.environ.get("CAM_PORT")
cam_attr = os.environ.get("CAM_CAPTURE_ATTR")
#for i in range(1,300):
capture_url = 'rtsp://admin:admin@'+cam_ip+':'+cam_port+"/"+cam_attr+".sdp"
cam = cv2.VideoCapture(0) #capture_url
# Список цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]


# Функция для обработки изображения
def process_image(image, tracks):
    # Загрузка изображения
    results = model(image)[0]

    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    boxes_id = results.boxes.id
    # Подготовка словаря для группировки результатов по классам
    grouped_objects = {}
    grouped_objects["person"] = []
    index = 0;
    # Рисование рамок и группировка результатов
    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]  # Выбор цвета для класса
        if class_name != "person":
            continue
        grouped_objects[class_name].append(box)

        # Рисование рамок на изображении
        x1, y1, x2, y2 = box
        centre = (int((x1 + x2)/2), int((y1 + y2)/2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if len(tracks) <= index:
            tracks.append([])
        tracks[index].append(centre)
        index += 1
    return image

    # Сохранение данных в текстовый файл

def draw_tracks(image, tracks):
    for track in tracks:
        for point_index in range(1, len(track)):
            pt1 = track[point_index - 1]
            pt2 = track[point_index]
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    return image

tracks = []
while True:
    ret, frame = cam.read()
    if not ret:
        break
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    frame = process_image(frame, tracks)
    frame = draw_tracks(frame, tracks)
    cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"): break
cam.release()