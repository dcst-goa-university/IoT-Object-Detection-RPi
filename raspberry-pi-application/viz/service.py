import cv2

class VisualizeService:
    def __init__(self, conf_thresh=0.4, min_box_size=32):
        self.conf_thresh = conf_thresh
        self.min_box_size = min_box_size

    def draw_boxes(self, frame, detections):
        annotated = detections.plot()  # YOLO’s own visualization
        return annotated
