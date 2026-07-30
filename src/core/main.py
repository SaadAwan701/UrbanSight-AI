import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("video.mp4")

while True:
    success, frame = cap.read()
    if success == False:
        break

    results = model(frame, conf=0.25, classes=[0, 2], imgsz=640)[0]
    # annotated_frame = results[0].plot()
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = []
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        class_name = model.names[class_id]
        labels.append(f"{class_name} #{tracker_id}")

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    cv2.imshow("annotated frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
