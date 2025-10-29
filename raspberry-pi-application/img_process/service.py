import cv2


class ImageProcessService:
    def __init__(self):
        pass

    def resize_image(self, frame, size=(640, 480)):
        # Example image processing: convert to grayscale
        img = cv2.resize(frame, size)
        return img
