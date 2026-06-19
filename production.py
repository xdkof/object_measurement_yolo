import cv2
from ultralytics import YOLO


BELT_START_Y_PIXEL = 150  
BELT_END_Y_PIXEL = 680    

VISIBLE_BELT_LENGTH_MM = 1945.0  
TARGET_LENGTH_MM = 1200.0  
TOLERANCE_MM = 15.0        
MIN_VALID_LENGTH_MM = 1000.0

def main():
    model = YOLO('best.pt')
    video_path = "videos/steel_plates_cutting.mp4"
    cap = cv2.VideoCapture(video_path)

    max_recorded_lengths = {}
    completed_ids = set()
    assigned_production_ids = {} 
    production_sheet_counter = 1
    
    belt_pixel_length = BELT_END_Y_PIXEL - BELT_START_Y_PIXEL

    print("--- Production QC Stream Active (Forced Finalization Enabled) ---")

    def finalize_sheet(tid):
        nonlocal production_sheet_counter

        if tid in completed_ids:
            return

        final_length = max_recorded_lengths[tid]
        if final_length < MIN_VALID_LENGTH_MM:
            return

        completed_ids.add(tid)
        official_id = production_sheet_counter
        assigned_production_ids[tid] = official_id
        production_sheet_counter += 1

        status = "PASS" if abs(final_length - TARGET_LENGTH_MM) <= TOLERANCE_MM else "REJECT"
        print(f"[VERDICT] Sheet #{official_id} | Status: {status} | Final Length: {final_length:.1f} mm")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)[0]
        current_frame_ids = []

        if results.boxes and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            current_frame_ids = track_ids.tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                sheet_pixel_length = y2 - y1
                current_length_mm = (sheet_pixel_length / belt_pixel_length) * VISIBLE_BELT_LENGTH_MM
                
                if track_id not in max_recorded_lengths:
                    max_recorded_lengths[track_id] = current_length_mm
                elif current_length_mm > max_recorded_lengths[track_id]:
                    max_recorded_lengths[track_id] = current_length_mm

                # UI
                if track_id in assigned_production_ids:
                    official_id = assigned_production_ids[track_id]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Sheet #{official_id} | Final: {max_recorded_lengths[track_id]:.1f} mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (0, 255, 0) if abs(max_recorded_lengths[track_id] - TARGET_LENGTH_MM) <= TOLERANCE_MM else (255, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Measuring: {max_recorded_lengths[track_id]:.1f} mm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for tid in list(max_recorded_lengths.keys()):
            if tid not in current_frame_ids and tid not in completed_ids:
                finalize_sheet(tid)

        cv2.imshow("QC Operator Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for tid in list(max_recorded_lengths.keys()):
        finalize_sheet(tid)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
