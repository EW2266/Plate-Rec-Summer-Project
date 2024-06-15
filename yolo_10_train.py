from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    model.train(data="datasets/data.yaml", epochs=500, imgsz=416, save=True, patience=0, batch=32, resume=True)