from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

from model.service import ModelService
from img_process.service import ImageProcessService
from viz.service import VisualizeService
from streamio.service import StreamIOService


modelService = ModelService("yolov8n.pt")
imageProcessService = ImageProcessService()
visualizeService = VisualizeService()
streamIOService = StreamIOService()

app = FastAPI()

def frame_generator():
    while True:
        frame = streamIOService.get_frame()
        resized = imageProcessService.resize_image(frame)
        detections = modelService.infer(resized)
        annotated = visualizeService.draw_boxes(resized, detections)

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/stream")
def stream_endpoint():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

