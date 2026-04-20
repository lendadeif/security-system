import cv2
import time
import threading
import numpy as np
from loguru import logger
from camera   import Camera
from detector import PersonDetector
from FaceID   import FaceID

VERIFY_COOLDOWN_FRAMES = 12     
TRACK_MAX_MISS         = 8
IOU_MATCH_THRESH       = 0.30   
UNAUTH_COOLDOWN        = 10

VERIFY_PADDING         = 60

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


class Track:

    _id_counter = 0

    def __init__(self, bbox):
        Track._id_counter  += 1
        self.id             = Track._id_counter
        self.bbox           = bbox
        self.result         = {"authorized": None, "name": "...", "score": 0.0}
        self.miss           = 0
        self.frames_since_verify = VERIFY_COOLDOWN_FRAMES  
        self.verifying      = False   

    def update(self, bbox):
        self.bbox  = bbox
        self.miss  = 0

    @property
    def track_key(self) -> str:
        return str(self.id)

class SecuritySystem:

    def __init__(self):
        logger.info("Initializing Security System...")
        self.camera   = Camera()
        self.detector = PersonDetector()
        self.face_id  = FaceID()

        self._tracks        = []              
        self._tracks_lock   = threading.Lock()
        self._frame_counter = 0
        self._alert_times   = {}

        logger.info("System ready.")

    def run(self):
        logger.info("Starting — press Q to quit")
        self.detector.load()
        self.camera.start()

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                self._frame_counter += 1
                detections = self.detector.detect(frame)   
                with self._tracks_lock:
                    self._update_tracks(detections)
                    live_tracks = list(self._tracks)

                for track in live_tracks:
                    if (not track.verifying
                            and track.frames_since_verify >= VERIFY_COOLDOWN_FRAMES):
                        track.verifying = True
                        threading.Thread(
                            target  = self._verify_track,
                            args    = (frame.copy(), track),
                            daemon  = True
                        ).start()

                
                with self._tracks_lock:
                    for t in self._tracks:
                        t.frames_since_verify += 1

                
                for track in live_tracks:
                    if track.result.get("authorized") is False:
                        self._handle_unauthorized(frame, track)

                
                display = self._draw(frame, live_tracks)
                cv2.putText(display, f"FPS: {self.camera.get_fps()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (180, 180, 180), 2)

                cv2.imshow("Security System", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("Stopped.")

    def _update_tracks(self, detections):
       
        matched_track_ids = set()
        matched_det_idxs  = set()

        if self._tracks and detections:
            for di, det in enumerate(detections):
                best_iou   = IOU_MATCH_THRESH
                best_track = None
                for track in self._tracks:
                    if track.id in matched_track_ids:
                        continue
                    score = iou(track.bbox, det["bbox"])
                    if score > best_iou:
                        best_iou   = score
                        best_track = track

                if best_track is not None:
                    best_track.update(det["bbox"])
                    matched_track_ids.add(best_track.id)
                    matched_det_idxs.add(di)

        for track in self._tracks:
            if track.id not in matched_track_ids:
                track.miss += 1

        dead = [t for t in self._tracks if t.miss > TRACK_MAX_MISS]
        for t in dead:
            self.face_id.clear_track(t.track_key)
        self._tracks = [t for t in self._tracks if t.miss <= TRACK_MAX_MISS]

        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                self._tracks.append(Track(det["bbox"]))


    def _verify_track(self, frame, track: Track):

        x1, y1, x2, y2 = track.bbox
        h, w            = frame.shape[:2]

        x1c = max(0, x1 - VERIFY_PADDING)
        y1c = max(0, y1 - VERIFY_PADDING)
        x2c = min(w, x2 + VERIFY_PADDING)
        # Top half only — face region
        y2c = min(h, y1 + (y2 - y1) // 2 + VERIFY_PADDING)

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            track.verifying = False
            return

        result = self.face_id.verify(crop, track_id=track.track_key)

        with self._tracks_lock:
            # Make sure the track still exists (not culled during verify)
            if any(t.id == track.id for t in self._tracks):
                track.result = result
                track.frames_since_verify = 0

        track.verifying = False


    def _handle_unauthorized(self, frame, track: Track):
        now  = time.time()
        name = track.result.get("name", "Unknown")
        key  = f"{track.id}:{name}"

        if now - self._alert_times.get(key, 0) < UNAUTH_COOLDOWN:
            return

        self._alert_times[key] = now

        x1, y1, x2, y2 = track.bbox
        path = f"captures/unauthorized_{int(now)}.jpg"
        cv2.imwrite(path, frame[y1:y2, x1:x2])

        logger.warning(f"UNAUTHORIZED — {path}")
        print(f"\n{'='*50}")
        print(f"  ALERT: UNAUTHORIZED PERSON  (track #{track.id})")
        print(f"  {path}")
        print(f"  {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}\n")


    def _draw(self, frame, tracks) -> np.ndarray:
        display = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            result     = track.result
            authorized = result.get("authorized")
            name       = result.get("name",  "...")
            score      = result.get("score", 0.0)

            if authorized is None:
                color = (0, 165, 255)    # Orange — identifying
            elif authorized:
                color = (0, 220, 100)    # Green  — authorized
            else:
                color = (0, 0, 255)      # Red    — unauthorized

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if authorized is None:
                label = f"#{track.id} Identifying..."
            elif authorized:
                label = f"#{track.id} {name}  {score:.2f}"
            else:
                label = f"#{track.id} UNAUTHORIZED  {score:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(display,
                (x1, y1 - th - 12), (x1 + tw + 10, y1),
                color, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

        return display


if __name__ == "__main__":
    system = SecuritySystem()
    system.run()
