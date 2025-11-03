import cv2
import numpy as np

class ImageProcessService:
    def resize_image(self, frame: np.ndarray, size=(640, 640)) -> np.ndarray:
        return cv2.resize(frame, size)
