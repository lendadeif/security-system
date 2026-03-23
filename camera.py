import cv2
import time
import threading
from loguru import logger
from config import (
    CAMERA_SOURCE, CAMERA_WIDTH,
    CAMERA_HEIGHT, CAMERA_FPS
)


class Camera:
    def __init__(self):
        self.source = CAMERA_SOURCE
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.fps = CAMERA_FPS

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._start_time = None

    def start(self):
        logger.info(f"Opening camera source: {self.source}")
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.source}. "
                "Check that your webcam is connected and not used by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Camera ready — resolution: {actual_w}x{actual_h} @ {self.fps}fps")

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read(self):

        with self._lock:
            return cv2.flip(self._frame.copy(), 1) if self._frame is not None else None

    def stop(self):
        """Stop capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped.")

    def get_fps(self):
        """Return the real measured FPS since start."""
        if not self._start_time or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return round(self._frame_count / elapsed, 1)

    def is_running(self):
        return self._running

    def _capture_loop(self):
        """Background thread: continuously grab frames from the camera."""
        logger.info("Capture thread started.")
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to grab frame — retrying...")
                time.sleep(0.1)
                continue

            with self._lock:
                self._frame = frame
            self._frame_count += 1

        logger.info("Capture thread stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
