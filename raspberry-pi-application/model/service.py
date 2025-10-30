from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime
import cv2


class ModelService:
    def __init__(self, repo_id: str, file_name: str, revision: str):
        model_path = self._download_model(repo_id, file_name, revision)
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def _download_model(self, repo_id: str, file_name: str, revision: str) -> str:
        return hf_hub_download(repo_id=repo_id, filename=file_name, revision=revision)

    def infer(self, tensor: np.ndarray, conf_thresh: float = 0.0) -> np.ndarray:
        """
        Run ONNX inference and return detections as [x1, y1, x2, y2, conf, cls].
        Works with YOLOv8 ONNX models outputting (84, 8400).
        """
        preds = self.session.run(None, {"images": tensor})[0]
        preds = np.squeeze(preds)

        # Transpose if output shape is (84, 8400)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # → (8400, 84)

        boxes = []
        for det in preds:
            x, y, w, h, obj_conf, *cls_scores = det

            # Confidence computation
            cls = np.argmax(cls_scores)
            cls_conf = cls_scores[cls]
            conf = obj_conf * cls_conf
            if conf < conf_thresh:
                continue

            # Convert from center xywh → corner xyxy
            x1, y1, x2, y2 = (
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2,
            )

            # Scale normalized coordinates (0–1) to 640x640 image size
            if max(x2, y2) <= 1.5:
                x1, y1, x2, y2 = [v * 640 for v in (x1, y1, x2, y2)]

            boxes.append([x1, y1, x2, y2, conf, cls])

        boxes = np.array(boxes)
        print(f"✅ {len(boxes)} detections after filtering")
        return boxes
