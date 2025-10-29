import cv2
import numpy as np
class ImageProcessService:
    def __init__(self):
        pass

    def resize_image(self, frame, size=(640, 480)):
        # Example image processing: convert to grayscale
        img = cv2.resize(frame, size)
        return img

    def to_numpy_tensor(self, frame):
        """
        Convert an OpenCV BGR image to a normalized NumPy tensor
        ready for model inference (1, 3, H, W).
        """
        # Convert BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img 
