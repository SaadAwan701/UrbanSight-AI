import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


cap = cv2.VideoCapture("src/core/video.mp4")
if not cap.isOpened():
    print("ERROR: OpenCV could not find or open the video file!")
    exit()


# Update this path!
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"DEBUG - Video Resolution: {width}x{height}")

# 3. Calculate the Y-coordinate for the bottom 30% of the screen
start_y = int(height * 0.70)

# 4. Draw the dynamic polygon
polygon = np.array([[0, start_y], [width, start_y], [width, height], [0, height]])

# 5. Initialize without the deprecated parameter, but add thickness!
zone = sv.PolygonZone(polygon=polygon)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=6)

entered_vehicles = set()
while True:
    success, frame = cap.read()
    if success == False:
        break

    results = model(frame, conf=0.25, classes=[0, 2], imgsz=640)[0]
    # annotated_frame = results[0].plot()
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    zone_mask = zone.trigger(detections=detections)
    cars_in_zone = detections[zone_mask]

    # 3. Loop through the IDs of those trespassing cars
    for tracker_id in cars_in_zone.tracker_id:
        if tracker_id not in entered_vehicles:
            # First time seeing this car in the zone!
            entered_vehicles.add(tracker_id)
            print(f"🚨 ALERT: Vehicle #{tracker_id} entered the restricted zone!")
    # 2. Count how many 'True' values we got
    trespassers = len(detections[zone_mask])

    # 3. Print the alert to the terminal
    if trespassers > 0:
        print(f"ALERT: {trespassers} vehicle(s) inside the restricted zone!")

    labels = []
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        class_name = model.names[class_id]
        labels.append(f"{class_name} #{tracker_id}")

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    frame = zone_annotator.annotate(scene=frame)

    cv2.imshow("annotated frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
