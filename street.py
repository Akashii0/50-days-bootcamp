import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2. imread('people.jpg')
result = model (image)

for i,r in enumerate(result):
    detection = r.boxes.data.tolist()
    #print(r.names[8])
    #print(r)

    names = r.names
    classes = r.boxes.cls.tolist()

for Labels, detection in zip(classes, detection):
    Labels = names[Labels]
    print (detection)