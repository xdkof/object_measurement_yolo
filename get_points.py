import cv2


points = []


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point {len(points)} captured: [{x}, {y}]")
        

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Conveyor Frame', frame)


video_path = "videos/steel_plates_cutting.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not load video frame.")
    exit()

print("INSTRUCTIONS:")
print("1. Click 4 points on the conveyor belt that form a perfect physical rectangle.")
print("2. Click them in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
print("3. Press any key on your keyboard to close the window when finished.")


cv2.imshow('Conveyor Frame', frame)
cv2.setMouseCallback('Conveyor Frame', click_event)


cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n--- COPY THESE 4 POINTS ---")
print(f"SOURCE_POINTS = {points}")