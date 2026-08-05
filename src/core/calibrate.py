import cv2

# 1. Grab the first frame of your video
cap = cv2.VideoCapture("src/core/video.mp4")  # Ensure this path is correct
success, frame = cap.read()
cap.release()

if not success:
    print("Could not read video.")
    exit()

points = []


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point recorded: [{x}, {y}]")

        # Draw a dot where you clicked
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw lines connecting the dots
        if len(points) >= 2:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)


print("Click 4 points on the road in this exact order:")
print("1. Top-Left")
print("2. Top-Right")
print("3. Bottom-Right")
print("4. Bottom-Left")
print("Press any key to close the window when done.")

cv2.imshow("Calibration", frame)
cv2.setMouseCallback("Calibration", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n--- COPY THIS INTO main.py ---")
print(f"SOURCE_POINTS = np.array({points})")
