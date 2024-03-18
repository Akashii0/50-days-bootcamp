import cv2
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov8n.pt')

# Load video file or capture device
img = cv2.imread("people.jpg")  
resize = cv2.resize(img, dsize=(600,400))
# Detect objects
results = model.predict(resize, conf=0.5, classes=[0])

count = 0

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cv2.rectangle(resize, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(resize, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    count += 1

print("there are " + str(count)+ " people in the image")

# Display output
cv2.imshow("Akeem's YOLOv8 Person Detection", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()