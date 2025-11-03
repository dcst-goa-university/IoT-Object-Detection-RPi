from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import time
import threading
from collections import deque

from model.service import ModelService
from img_process.service import ImageProcessService
from viz.service import VisualizeService
from streamio.service import StreamIOService


# Initialize services
modelService = ModelService("yolov8n.pt")
imageProcessService = ImageProcessService()
visualizeService = VisualizeService()
streamIOService = StreamIOService()

app = FastAPI()

# Shared buffer for preprocessed & annotated frames
frame_buffer = deque(maxlen=30)  # holds up to 30 frames (adjust for smoothness)
stop_streaming = False


def producer():
    """Continuously grab frames, process them, and push to buffer."""
    global stop_streaming

    # Warm-up phase: collect frames for 5 seconds
    print("🕒 Warming up for 5 seconds...")
    start_time = time.time()
    while time.time() - start_time < 5:
        frame = streamIOService.get_frame()
        resized = imageProcessService.resize_image(frame)
        detections = modelService.infer(resized)
        annotated = visualizeService.draw_boxes(resized, detections)
        frame_buffer.append(annotated)
    print("✅ Warm-up done, streaming started.")

    # Continuous loop
    while not stop_streaming:
        frame = streamIOService.get_frame()
        resized = imageProcessService.resize_image(frame)
        detections = modelService.infer(resized)
        annotated = visualizeService.draw_boxes(resized, detections)
        frame_buffer.append(annotated)


def frame_generator():
    """Stream frames from the buffer."""
    # Start background thread for continuous frame processing
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    while True:
        if not frame_buffer:
            time.sleep(0.01)
            continue

        frame = frame_buffer[-1]  # take latest frame for lowest latency
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/stream")
def stream_endpoint():
    """Streaming endpoint for MJPEG."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("shutdown")
def shutdown_event():
    global stop_streaming
    stop_streaming = True
