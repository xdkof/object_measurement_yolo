from ultralytics import YOLO

def main():
    # 1. Load a pre-trained "nano" model (lightweight, fast, and great for prototyping)
    model = YOLO('yolov8n.pt')

    # 2. Train the model on your custom dataset
    # We point directly to the data.yaml file inside your new dataset folder
    results = model.train(
        data='dataset/data.yaml',   # Path to your configuration file
        epochs=50,                  # Number of training passes through the dataset
        imgsz=640,                  # Image size (standard for YOLOv8)
        batch=8,                    # Number of images processed per batch (adjust based on your RAM/VRAM)
        device='cpu',               # Set to 'cpu' by default; change to 0 if you have an NVIDIA GPU configured
        workers=2                   # Dataloader workers (keeps it stable on standard setups)
    )

if __name__ == '__main__':
    main()