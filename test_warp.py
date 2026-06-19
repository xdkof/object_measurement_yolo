import cv2
import numpy as np


SOURCE_POINTS = np.float32([ [429, 162], [700, 127], [819, 229], [345, 273] ])

width = 500
height = 1000


DEST_POINTS = np.float32([
    [0, 0],               # Top-Left
    [width, 0],           # Top-Right
    [width, height],      # Bottom-Right
    [0, height]           # Bottom-Left
])


matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DEST_POINTS)

def main():
    video_path = "videos/steel_plates_cutting.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not load frame.")
        return

    warped_frame = cv2.warpPerspective(frame, matrix, (width, height))

    cv2.imshow("Original Tilted View", frame)
    cv2.imshow("Flattened Bird's-Eye View", warped_frame)
    
    print("Press any key on your keyboard to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()