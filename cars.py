import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video file or capture device
img = cv2.imread("car_test_1.jpg")  
resize = cv2.resize(img, dsize=(600,400))


count = 0
result = model(resize)

for i,r in enumerate(result):
    detection = r.boxes.data.tolist()

    classes = r.boxes.cls.tolist()
    names = r.names
    car_object = 2.0
    print(classes)
    print(names)
    for labels, detected_weapon in zip(classes,detection):
        if labels == car_object:
            count += 1
            x,y,w,h, conf, _ = detected_weapon
            labels = names[labels]
            cv2.putText(resize, labels, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            cv2.putText(resize, str(round(conf, 2)), (int(x), int(h)), cv2.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            # cv2.putText(resize, labels, org=(int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=[50,50,255], thickness=1)
            # cv2.putText(resize, str(round(conf,2)), org=(int(x), int(h)), cv2.FONT_HERSHEY_DUPLEX, fontScale= 0.5, color=[50,50,255],thickness=1)
if count == 0:
    print("there is no car")

elif count == 1:
    print("there is a car")

else:
    print("there are" + str(count)+ ' cars')


# Display output
cv2.imshow("Akeem's YOLOv8 car Detection", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

