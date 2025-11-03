from ultralytics import YOLO

class ModelService:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def infer(self, frame):
        results = self.model(frame)
        return results[0]  # single image inference
