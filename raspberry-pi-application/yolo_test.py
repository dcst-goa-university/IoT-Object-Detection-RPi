from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt 
from pathlib import Path
import shutil

def main():
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "yolov8n.pt"

    if not model_path.exists():
        print("Model not found locally. Downloading yolov8n.pt ...")
        temp_model = YOLO("yolov8n.pt")  # downloads to cache
        # Copy from cache to model folder
        shutil.copy(temp_model.ckpt_path, model_path)
        model = temp_model
    else:
        model = YOLO(str(model_path))

    image_path = base_dir / 'output_with_boxes.jpg'

    print("Running inference...")
    results = model(image_path)
    result = results[0]

    # Visualize
    annotated = result.plot()
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLOv8 Inference Result")
    plt.show()

    print("\nDetected objects:")
    for box in result.boxes:
        cls = int(box.cls)
        name = result.names[cls]
        conf = float(box.conf)
        print(f" - {name}: {conf:.2f}")

if __name__ == "__main__":
    main()

