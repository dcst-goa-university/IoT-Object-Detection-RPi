from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from multiprocessing import Process, Queue
import cv2
import time

from model.service import ModelService
from img_process.service import ImageProcessService
from viz.service import VisualizeService
from streamio.service import StreamIOService

app = FastAPI()

# Instantiate services (inside processes to avoid GPU/IO conflicts)
def streamio_process(input_queue):
    stream_service = StreamIOService()
    img_service = ImageProcessService()
    while True:
        frame = stream_service.get_frame()
        resized = img_service.resize_image(frame)
        if not input_queue.full():
            input_queue.put(resized)
        else:
            time.sleep(0.01)

def model_process(input_queue, output_queue):
    model_service = ModelService("yolov8n.pt")
    viz_service = VisualizeService()
    while True:
        if not input_queue.empty():
            frame = input_queue.get()
            detections = model_service.infer(frame)
            annotated = viz_service.draw_boxes(frame, detections)
            if not output_queue.full():
                output_queue.put(annotated)
        else:
            time.sleep(0.01)

# Shared queues
input_queue = Queue(maxsize=5)
output_queue = Queue(maxsize=5)

# Start background processes
p1 = Process(target=streamio_process, args=(input_queue,), daemon=True)
p2 = Process(target=model_process, args=(input_queue, output_queue), daemon=True)
p1.start()
p2.start()

def frame_generator():
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            time.sleep(0.01)

@app.get("/stream")
def stream_endpoint():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
