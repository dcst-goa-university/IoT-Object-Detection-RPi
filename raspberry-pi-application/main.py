import cv2
import numpy as np
from model.service import ModelService
from img_process.service import ImageProcessService
from viz.service import VisualizeService
from streamio.service import StreamIOService


def main():
    # Initialize services
    model = ModelService(
        repo_id="SpotLab/YOLOv8Detection",
        file_name="yolov8n.onnx",
        revision="3005c67"
    )
    image_proc = ImageProcessService()
    visualizer = VisualizeService(conf_thresh=0.4, min_box_size=32)
    stream = StreamIOService(camera_index=0)

    # Capture one frame
    frame = stream.get_frame()

    # Preprocess
    resized = image_proc.resize_image(frame, (640, 640))
    tensor = image_proc.to_numpy_tensor(resized)

    # Inference
    detections = model.infer(tensor)

    # Visualization
    output = visualizer.draw_boxes(frame, detections)

    # Display
    cv2.imshow("YOLOv8 Inference (1 Frame)", output)
    print("✅ Press any key to close window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    stream.release()


if __name__ == "__main__":
    main()
