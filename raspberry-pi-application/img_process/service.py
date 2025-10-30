import cv2
import numpy as np

class ImageProcessService:
    """Handles image preprocessing for inference (resize, normalize, tensor conversion)."""

    def __init__(self):
        pass

    def resize_image(self, frame: np.ndarray, size=(640, 640)) -> np.ndarray:
        """Resize the image to a target size."""
        return cv2.resize(frame, size)

    def bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR image to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1]."""
        return frame.astype(np.float32) / 255.0

    def to_chw(self, frame: np.ndarray) -> np.ndarray:
        """Convert HWC layout (OpenCV) to CHW layout (model)."""
        return np.transpose(frame, (2, 0, 1))

    def to_numpy_tensor(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert an OpenCV BGR image to a normalized NumPy tensor
        ready for model inference (1, 3, H, W).
        """
        rgb = self.bgr_to_rgb(frame)
        norm = self.normalize(rgb)
        chw = self.to_chw(norm)
        return np.expand_dims(chw, axis=0)
