import cv2
import os

# The empty output folder where your Python script will save the extracted .jpg images
output_folder = "dataset_frames"
os.makedirs(output_folder, exist_ok=True)

# The relative path to your video file
video_path = "videos/steel_plates_cutting.mp4" 
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0
interval = 3  # Extract every 3rd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Video is finished
    
    # Save the frame only if it hits our interval
    if frame_count % interval == 0:
        filename = os.path.join(output_folder, f"plate_frame_{saved_count}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        
    frame_count += 1

cap.release()
print(f"Done! Saved {saved_count} images to '{output_folder}'.")