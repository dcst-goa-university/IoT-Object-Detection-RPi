import cv2
import requests
import numpy as np
import threading
import time
from collections import deque


class StreamConsumer:
    def __init__(self, stream_url, max_buffer_time=5.0):
        self.stream_url = stream_url
        self.buffer = deque()
        self.stop_flag = False
        self.max_buffer_time = max_buffer_time  # in seconds
        self.buffer_lock = threading.Lock()

    def _fetch_stream(self):
        """Fetch MJPEG stream and decode frames."""
        print(f"Connecting to stream: {self.stream_url}")
        response = requests.get(self.stream_url, stream=True)
        bytes_data = b""

        for chunk in response.iter_content(chunk_size=1024):
            if self.stop_flag:
                break

            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                # Decode JPEG to image
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    with self.buffer_lock:
                        self.buffer.append((frame, time.time()))
                        # remove old frames (>2s)
                        while self.buffer and (time.time() - self.buffer[0][1]) > self.max_buffer_time:
                            self.buffer.popleft()

    def start(self):
        """Start stream fetching in background."""
        self.stop_flag = False
        self.thread = threading.Thread(target=self._fetch_stream, daemon=True)
        self.thread.start()
        print("Stream fetching started.")

    def stop(self):
        self.stop_flag = True
        self.thread.join()

    def display(self):
        """Display buffered frames smoothly."""
        print("Displaying stream... Press 'q' to quit.")
        while not self.stop_flag:
            frame = None
            with self.buffer_lock:
                if self.buffer:
                    frame = self.buffer.popleft()[0]

            if frame is not None:
                cv2.imshow("Live Stream", frame)

            # Display next frame every ~33 ms (~30 FPS)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change this to the server's IP address
    stream_url = "http://192.168.0.146:8000/stream"

    client = StreamConsumer(stream_url, max_buffer_time=2.0)
    client.start()
    client.display()
