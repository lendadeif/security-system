# main.py
import cv2
import time
import numpy as np
from loguru import logger
from camera   import Camera
from detector import PersonDetector
from FaceID  import FaceID


VERIFY_EVERY     = 10     
VERIFY_PADDING   = 60     
UNAUTH_COOLDOWN  = 10     
MATCH_THRESHOLD  = 0.6


class SecuritySystem:

    def __init__(self):
        logger.info("Initializing Security System...")
        self.camera   = Camera()
        self.detector = PersonDetector()
        self.face_id  = FaceID()

        self._last_results  = {}   
        self._frame_counter = 0

        self._alert_times = {}

        logger.info("System ready.")

    def run(self):
        logger.info("Starting security system — press Q to quit")

        self.detector.load()
        self.camera.start()

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                self._frame_counter += 1
                display = frame.copy()

                detections = self.detector.detect(frame)

                run_verify = (self._frame_counter % VERIFY_EVERY == 0)

                for i, det in enumerate(detections):
                    if run_verify:
                        result = self._verify_person(frame, det)
                        self._last_results[i] = result
                    else:
                        result = self._last_results.get(i, {
                            "authorized": None,
                            "name":       "...",
                            "score":      0.0
                        })

                    det["result"] = result

                    if result.get("authorized") is False:
                        self._handle_unauthorized(frame, det)

                display = self._draw(frame, detections)

                cv2.putText(
                    display,
                    f"FPS: {self.camera.get_fps()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 180, 180), 2
                )

                cv2.imshow("Security System", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("System stopped.")


    def _verify_person(self, frame, det) -> dict:
        x1, y1, x2, y2 = det["bbox"]
        h, w            = frame.shape[:2]

        x1c = max(0, x1 - VERIFY_PADDING)
        y1c = max(0, y1 - VERIFY_PADDING)
        x2c = min(w, x2 + VERIFY_PADDING)
        y2c = min(h, y2 + VERIFY_PADDING)

        crop = frame[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            return {"authorized": None, "name": "...", "score": 0.0}

        return self.face_id.verify(crop)


    def _handle_unauthorized(self, frame, det):
        name = det["result"].get("name", "Unknown")
        now  = time.time()

        last = self._alert_times.get(name, 0)
        if now - last < UNAUTH_COOLDOWN:
            return

        self._alert_times[name] = now

        logger.warning(f"UNAUTHORIZED PERSON DETECTED — {name}")

        x1, y1, x2, y2 = det["bbox"]
        snapshot_path   = f"captures/unauthorized_{int(now)}.jpg"
        cv2.imwrite(snapshot_path, frame)
        logger.info(f"Snapshot saved → {snapshot_path}")

        print(f"\n{'='*50}")
        print(f"ALERT: UNAUTHORIZED PERSON")
        print(f"Snapshot: {snapshot_path}")
        print(f"Time:     {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}\n")


    def _draw(self, frame, detections) -> np.ndarray:
        display = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            result          = det.get("result", {})
            authorized      = result.get("authorized")
            name            = result.get("name", "...")
            score           = result.get("score", 0.0)

            if authorized is None:
                color = (0, 165, 255)      
            elif authorized:
                color = (0, 220, 100)      
            else:
                color = (0, 0, 255)        

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = (
                f"{name}  {score:.2f}" if authorized else
                f"{score} UNAUTHORIZED"         if authorized is False else
                "Identifying..."
            )
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            cv2.rectangle(
                display,
                (x1, y1 - th - 12),
                (x1 + tw + 10, y1),
                color, -1
            )
            cv2.putText(
                display, label,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2
            )

        return display


if __name__ == "__main__":
    system = SecuritySystem()
    system.run()