import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load image
img = cv2.imread("car_test_1.jpg")
resize = cv2.resize(img, dsize=(600,400))
# Detect cars
results = model.predict(resize, conf=0.5, classes=[0])

# Initialize car count
car_count = 0

# Draw bounding boxes and labels on image and count number of cars
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cv2.rectangle(resize, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(resize, 'Car', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    car_count += 1

# Display output and print car count
cv2.imshow('YOLOv8 Car Detection', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Number of cars detected: {car_count}")