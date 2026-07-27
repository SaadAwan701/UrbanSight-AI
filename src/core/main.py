import cv2
from ultralytics import YOLO

model = YOLO("yolov8m-visdrone")
cap = cv2.VideoCapture("video.mp4")

while True:
    success, frame = cap.read()
    if success == False:
        break
    results = model(frame, conf=0.5, classes=[0, 2], imgsz=1280)
    annotated_frame = results[0].plot()

    cv2.imshow("annotated frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
