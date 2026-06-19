from ultralytics import YOLO

def main():

    model = YOLO('best.pt')

    video_path = "videos/steel_plates_cutting.mp4"

    results = model.predict(source=video_path, show=True, conf=0.5)

if __name__ == '__main__':
    main()