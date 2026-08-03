import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import sqlite3
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.spatial.perspective import PerspectiveTransformer

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# --- SPEED ESTIMATION SETUP ---
# 1. The 4 corners of the road in the video (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
SOURCE_POINTS = np.array([[200, 300], [440, 300], [640, 640], [0, 640]])

# 2. What those 4 corners represent in the real world (e.g., 20m wide by 40m long)
TARGET_WIDTH_METERS = 20
TARGET_HEIGHT_METERS = 40
TARGET_POINTS = np.array(
    [
        [0, 0],
        [TARGET_WIDTH_METERS, 0],
        [TARGET_WIDTH_METERS, TARGET_HEIGHT_METERS],
        [0, TARGET_HEIGHT_METERS],
    ]
)

transformer = PerspectiveTransformer(SOURCE_POINTS, TARGET_POINTS)

# Dictionary to remember where a car was 1 second ago
vehicle_history = {}
# ------------------------------

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


os.makedirs("database", exist_ok=True)

# 2. Route the connection into the folder
conn = sqlite3.connect("database/urbansight_logs.db")
cursor = conn.cursor()

# Create a table if it doesn't exist yet
cursor.execute("""
    CREATE TABLE IF NOT EXISTS geofence_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        vehicle_id INTEGER
    )
""")
conn.commit()
# ----------------------

entered_vehicles = set()
entered_vehicles = set()
while True:
    success, frame = cap.read()
    if success == False:
        break

    results = model(frame, conf=0.25, classes=[0, 2], imgsz=640)[0]
    # annotated_frame = results[0].plot()
    detections = sv.Detections.from_ultralytics(results)
    # ... ByteTrack update line ...
    detections = tracker.update_with_detections(detections)

    # Get the FPS of the video so we know how much time passes between frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Fallback just in case

    labels = []

    for bbox, tracker_id, class_id in zip(
        detections.xyxy, detections.tracker_id, detections.class_id
    ):
        # 1. Find the bottom-center point of the bounding box (where the tires hit the road)
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)

        # 2. Warp that pixel into our flattened bird's-eye map
        warped_x, warped_y = transformer.transform_point(center_x, bottom_y)

        speed_kmh = 0

        # 3. Calculate distance if we have seen this car before
        if tracker_id in vehicle_history:
            prev_x, prev_y = vehicle_history[tracker_id]

            # Pythagorean theorem to find distance in meters
            distance_meters = np.sqrt(
                (warped_x - prev_x) ** 2 + (warped_y - prev_y) ** 2
            )

            # Time elapsed between one frame (1 / FPS)
            time_seconds = 1 / fps

            # Speed = Distance / Time (Meters per Second)
            speed_mps = distance_meters / time_seconds

            # Convert to KM/H
            speed_kmh = int(speed_mps * 3.6)

        # 4. Update the history for the next frame
        vehicle_history[tracker_id] = (warped_x, warped_y)

        # 5. Add speed to the label!
        class_name = model.names[class_id]
        labels.append(f"#{tracker_id} {class_name} | {speed_kmh} km/h")

    # ... zone mask logic ...

    zone_mask = zone.trigger(detections=detections)
    cars_in_zone = detections[zone_mask]

    # 3. Loop through the IDs of those trespassing cars
    for tracker_id in cars_in_zone.tracker_id:
        if tracker_id not in entered_vehicles:
            entered_vehicles.add(tracker_id)

            # Get exact current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Print to terminal
            print(f"🚨 ALERT: Vehicle #{tracker_id} entered at {current_time}")

            # Save to database
            cursor.execute(
                """
                INSERT INTO geofence_alerts (timestamp, vehicle_id) 
                VALUES (?, ?)
            """,
                (current_time, tracker_id),
            )
            conn.commit()
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
conn.close()
cv2.destroyAllWindows()
