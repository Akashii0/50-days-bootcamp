import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video file or capture device
img = cv2.imread("office.jpg")  
resize = cv2.resize(img, dsize=(600,400))


count = 0
result = model(resize)

for i,r in enumerate(result):
    detection = r.boxes.data.tolist()

    classes = r.boxes.cls.tolist()
    names = r.names
    office_object = [24.0, 26.0, 27.0, 28.0, 41.0, 56.0, 57.0, 58.0, 60.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 72.0, 73.0, 74.0, 75.0, 76.0]
    print(classes)
    print(names)
    for labels, detected_weapon in zip(classes,detection):
        if labels in office_object:
            count += 1
            x,y,w,h, conf, _ = detected_weapon
            labels = names[labels]
            cv2.rectangle(resize, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(resize, labels, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            cv2.putText(resize, str(round(conf, 2)), (int(x), int(h)), cv2.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            # cv2.putText(resize, labels, org=(int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=[50,50,255], thickness=1)
            # cv2.putText(resize, str(round(conf,2)), org=(int(x), int(h)), cv2.FONT_HERSHEY_DUPLEX, fontScale= 0.5, color=[50,50,255],thickness=1)
if count == 0:
    print("there is no office tool")

elif count == 1:
    print("there is" +str(count)+ " one office tool")

else:
    print("there are " + str(count)+ 'loads of office tools')


# Display output
cv2.imshow("Akeem's YOLOv8 office Detection", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

