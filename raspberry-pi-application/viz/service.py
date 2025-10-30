import cv2
import numpy as np

class VisualizeService:
    """Service to draw bounding boxes and labels on an image."""

    def __init__(
        self,
        conf_thresh: float = 0.3,
        box_thickness: int = 10,
        font_scale: float = 0.5,
        min_box_size: int = 256,
    ):
        """
        Args:
            conf_thresh: Minimum confidence to display a box
            box_thickness: Border thickness for boxes
            font_scale: Font scale for labels
            max_detections: Max number of boxes to draw
            min_box_size: Skip boxes smaller than this (px)
        """
        self.conf_thresh = conf_thresh
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.min_box_size = min_box_size

    def draw_boxes(self, frame: np.ndarray, detections: np.ndarray, class_names=None):
        """
        Draw bounding boxes on a frame.
        Args:
            frame: Original BGR image (H, W, 3)
            detections: ndarray of shape (N, 4) or (N, 6)
            class_names: Optional list of class names
        Returns:
            Annotated frame
        """
        if detections is None or len(detections) == 0:
            return frame

        annotated = frame.copy()
        detections = np.squeeze(detections)

        h, w, _ = annotated.shape

        for det in detections:
            det = np.ravel(det)
            if len(det) == 4:
                x1, y1, x2, y2 = [float(v) for v in det]
                conf, cls = 1.0, 0
            elif len(det) >= 6:
                x1, y1, x2, y2, conf, cls = [float(v) for v in det[:6]]
                if conf < self.conf_thresh:
                    continue
            else:
                continue

            # Skip invalid or small boxes
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Label
            label = (
                f"{class_names[int(cls)] if class_names and int(cls) < len(class_names) else int(cls)} "
                f"{conf:.2f}"
            )

            # Draw bounding box
            cv2.rectangle(
                annotated, 
                (x1, y1), 
                (x2, y2), 
                (0, 255, 0), 
                self.box_thickness
            )

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 4)),
                (x1 + text_w + 4, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),
                1
            )

        return annotated
