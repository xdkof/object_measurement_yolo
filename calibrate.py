import cv2
import numpy as np
import math
from ultralytics import YOLO


SOURCE_POINTS = np.float32([ [433, 156], [704, 128], [1053, 526], [156, 596] ])
width = 500
height = 1000
DEST_POINTS = np.float32([[0, 0], [width, 0], [width, height], [0, height]])


matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DEST_POINTS)

def main():

    model = YOLO('best.pt')


    video_path = "videos/steel_plates_cutting.mp4"
    cap = cv2.VideoCapture(video_path)
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended without finding a plate.")
            break
            

        results = model(frame, verbose=False)[0]
        

        if len(results.boxes) > 0:

            box = results.boxes.xyxy[0].cpu().numpy()
            
            top_center = [(box[0] + box[2]) / 2, box[1]]
            bottom_center = [(box[0] + box[2]) / 2, box[3]]
            

            points_to_warp = np.float32([[ top_center, bottom_center ]])
            

            warped_points = cv2.perspectiveTransform(points_to_warp, matrix)
            

            p1 = warped_points[0][0] # Warped top center
            p2 = warped_points[0][1] # Warped bottom center
            

            pixel_length = math.dist(p1, p2)
            

            target_length_mm = 1200
            mm_per_pixel = target_length_mm / pixel_length
            
            print("\n--- CALIBRATION SUCCESSFUL ---")
            print(f"Plate Length in Bird's-Eye View: {pixel_length:.2f} pixels")
            print(f"Locked System Scale: 1 pixel = {mm_per_pixel:.4f} mm")
            break 

    cap.release()

if __name__ == '__main__':
    main()