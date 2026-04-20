import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from FaceMesh import FaceMesh
from deepface import DeepFace


class FaceID:

    TICK_COUNT    = 60
    CIRCLE_RADIUS = 180
    INITIAL_HOLD  = 25


    MIN_MAGNITUDE = 0.20          
    YAW_MAX       = 35.0
    PITCH_MAX     = 25.0
    YAW_OFFSET    = 14.0
    TICK_SPREAD   = 3             
    LOGIN_THRESHOLD  = 0.55      
    MATCH_THRESHOLD  = 0.55
    VERIFY_THRESHOLD = 0.55

    EMBED_BUFFER_SIZE = 8

    MIN_CROP_SIZE  = 60           
    MIN_LAPLACIAN  = 40.0        

    def __init__(self):
        self.mesh          = FaceMesh()
        self._profile_path = Path("database/face_id_profiles.pkl")
        self._profiles     = self._load_profiles()
        self._embed_buffer = {}   

    def enroll(self, name: str) -> bool:
        cap          = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        initial_hold = 0
        started      = False
        captures     = []
        visited_angles = set()
        current_tick   = -1
        current_mag    = 0.0

        print(f"\nEnrolling: {name}")
        print("  Place your face in the circle, then slowly move your head.")
        print("  Press ESC to cancel.\n")

        while len(visited_angles) < self.TICK_COUNT:
            ret, frame = cap.read()
            if not ret:
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            cx, cy = w // 2, h // 2

            try:
                _, landmarks   = self.mesh.detect(frame, Draw=False)
                face_detected  = True
                face_in_circle = self._is_face_in_circle(landmarks, cx, cy)
            except Exception:
                face_detected  = False
                face_in_circle = False
                landmarks      = []

            yaw = pitch = roll = 0.0
            if face_in_circle and landmarks:
                yaw, pitch, roll          = self._estimate_pose(landmarks, frame.shape)
                current_tick, current_mag = self._pose_to_tick(yaw, pitch)

            if not started:
                if face_in_circle:
                    initial_hold += 1
                else:
                    initial_hold = max(0, initial_hold - 1)

                if (face_in_circle and landmarks
                        and initial_hold == self.INITIAL_HOLD // 2):
                    vec = self._embed_from_frame(frame, landmarks, w, h)
                    if vec is not None:
                        captures.append(vec)
                        print("  📸 Center captured!")

                if initial_hold >= self.INITIAL_HOLD:
                    started = True
                    print("Started — slowly move your head in a full circle!")

            elif face_in_circle and landmarks:
                if current_mag >= self.MIN_MAGNITUDE:
                    prev_count = len(visited_angles)

                    for offset in range(-self.TICK_SPREAD, self.TICK_SPREAD + 1):
                        visited_angles.add(
                            (current_tick + offset) % self.TICK_COUNT
                        )

                    if len(visited_angles) > prev_count:
                        vec = self._embed_from_frame(frame, landmarks, w, h,
                                                    quality_check=True)
                        if vec is not None:
                            captures.append(vec)
                            print(
                                f"  {len(visited_angles)}/{self.TICK_COUNT} "
                                f"ticks  yaw={yaw:+.1f}  pitch={pitch:+.1f}"
                            )
                        else:
                            for offset in range(-self.TICK_SPREAD, self.TICK_SPREAD + 1):
                                visited_angles.discard(
                                    (current_tick + offset) % self.TICK_COUNT
                                )

            display = self._build_frame(frame, cx, cy)
            self._draw_ticks(
                display, cx, cy, visited_angles,
                current_tick if started and current_mag >= self.MIN_MAGNITUDE
                else -1
            )

            if not face_detected or not face_in_circle:
                msg = "Position your face in the circle"
            elif not started:
                pct = int((initial_hold / self.INITIAL_HOLD) * 100)
                msg = f"Hold still...  {pct}%"
            else:
                msg = "Slowly move your head to\ncomplete the circle."

            self._draw_text(display, cx, cy, msg)
            cv2.imshow("Face ID", display)

            if cv2.waitKey(1) & 0xFF == 27:
                print("\nCancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return False

        self._animate_completion(cap, frame, cx, cy)
        cap.release()
        cv2.destroyAllWindows()

        normalised = [v / (np.linalg.norm(v) + 1e-9) for v in captures]
        self._profiles[name] = {"name": name, "vectors": normalised}
        self._save_profiles()
        print(f"\n🎉 '{name}' enrolled with {len(captures)} captures!")
        return True


    def login(self) -> dict:
        if not self._profiles:
            print("No enrolled users.")
            return {"success": False, "name": None, "score": 0.0}

        cap         = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        start_time  = time.time()
        TIMEOUT     = 15
        CONSEC_NEED = 5
        consecutive = 0
        scan_angle  = 0
        self._embed_buffer = {"login": []}

        print("\nFace ID — look at the camera...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT:
                cap.release()
                cv2.destroyAllWindows()
                print("Timed out.")
                return {"success": False, "name": None, "score": 0.0}

            ret, frame = cap.read()
            if not ret:
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            cx, cy = w // 2, h // 2

            try:
                _, landmarks   = self.mesh.detect(frame, Draw=False)
                face_detected  = True
                face_in_circle = self._is_face_in_circle(landmarks, cx, cy)
            except Exception:
                face_detected  = False
                face_in_circle = False
                landmarks      = []
                consecutive    = 0
                self._embed_buffer["login"] = []

            best_name, best_score = "Unknown", 0.0

            if face_in_circle and landmarks:
                vec = self._get_stable_embedding(frame, landmarks, w, h,
                                                 track_id="login")
                if vec is not None:
                    best_name, best_score = self._match(vec)
                    if best_score >= self.LOGIN_THRESHOLD:
                        consecutive += 1
                    else:
                        consecutive = 0

            if consecutive >= CONSEC_NEED:
                display = self._build_frame(frame, cx, cy)
                self._draw_ticks(display, cx, cy,
                                 set(range(self.TICK_COUNT)), -1)
                self._animate_success(cap, frame, cx, cy, best_name)
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nWelcome, {best_name}!  score={best_score:.4f}")
                return {"success": True, "name": best_name, "score": best_score}

            scan_visited = set()
            for offset in range(-4, 5):
                scan_visited.add((scan_angle + offset) % self.TICK_COUNT)
            scan_angle = (scan_angle + 2) % self.TICK_COUNT

            display = self._build_frame(frame, cx, cy)
            self._draw_ticks(display, cx, cy, scan_visited, scan_angle)

            if not face_detected or not face_in_circle:
                msg = "Look at the camera"
            elif best_score >= self.LOGIN_THRESHOLD:
                msg = f"Verifying...  {best_score:.2f}"
            else:
                msg = "Scanning..."

            self._draw_text(display, cx, cy, msg)
            self._draw_timeout_bar(display, w, elapsed, TIMEOUT)

            cv2.imshow("Face ID", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return {"success": False, "name": None, "score": 0.0}


    def verify(self, frame: np.ndarray, track_id: str = "default") -> dict:
        
        if not self._profiles:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        try:
            _, landmarks = self.mesh.detect(frame, Draw=False)
        except Exception:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        if not landmarks:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        h, w = frame.shape[:2]

        vec = self._get_stable_embedding(frame, landmarks, w, h,
                                         track_id=track_id)
        if vec is None:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        name, score = self._match(vec)
        authorized  = score >= self.VERIFY_THRESHOLD

        return {
            "authorized": authorized,
            "name":       name if authorized else "Unknown",
            "score":      score
        }

    def clear_track(self, track_id: str):
        """Call this when a track disappears to free memory."""
        self._embed_buffer.pop(track_id, None)


    def _embed_from_frame(self, frame, landmarks, w, h,
                          quality_check: bool = False) -> np.ndarray | None:

        face_crop = self._crop_face(frame, landmarks, w, h)
        if face_crop is None:
            return None

        if quality_check and not self._is_sharp(face_crop):
            return None

        return self._landmarks_to_vector(face_crop)

    def _get_stable_embedding(self, frame, landmarks, w, h,
                            track_id: str = "default") -> np.ndarray | None:

        face_crop = self._crop_face(frame, landmarks, w, h)
        if face_crop is None:
            return None

        vec = self._landmarks_to_vector(face_crop)
        if vec is None:
            return None

        if track_id not in self._embed_buffer:
            self._embed_buffer[track_id] = []

        buf = self._embed_buffer[track_id]
        buf.append(vec)
        if len(buf) > self.EMBED_BUFFER_SIZE:
            buf.pop(0)

        avg_vec = np.mean(buf, axis=0).astype(np.float32)
        norm    = np.linalg.norm(avg_vec)
        if norm < 1e-9:
            return None

        return avg_vec / norm

    def _crop_face(self, frame, landmarks, w, h, pad_ratio=0.18) -> np.ndarray | None:

        if not landmarks:
            return None

        xs = [lm[1] for lm in landmarks]
        ys = [lm[2] for lm in landmarks]

        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x  = int(face_w * pad_ratio)
        pad_y  = int(face_h * pad_ratio)

        x1 = max(0, int(min(xs)) - pad_x)
        y1 = max(0, int(min(ys)) - pad_y)
        x2 = min(w, int(max(xs)) + pad_x)
        y2 = min(h, int(max(ys)) + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] < self.MIN_CROP_SIZE or crop.shape[1] < self.MIN_CROP_SIZE:
            return None

        crop_bgr  = cv2.resize(crop, (112, 112))
        lab       = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _is_sharp(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= self.MIN_LAPLACIAN

    def _landmarks_to_vector(self, face_img: np.ndarray) -> np.ndarray | None:

        try:
            rep = DeepFace.represent(
                img_path          = face_img,
                model_name        = "ArcFace",
                enforce_detection = False
            )

            if not rep or len(rep) == 0:
                return None

            emb = rep[0].get("embedding", None)
            if emb is None or len(emb) == 0:
                return None

            vec  = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                return None

            return vec / norm

        except Exception:
            return None

    def _cosine_similarity(self, a, b) -> float:
        return float(np.dot(a, b))

    def _match(self, vec) -> tuple:
        
        if vec is None:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", 0.0

        for name, profile in self._profiles.items():
            vecs = profile["vectors"]
            if not vecs:
                continue

            scores = [self._cosine_similarity(vec, v) for v in vecs]

            k   = max(3, min(10, len(scores) // 10 + 1))
            top = sorted(scores, reverse=True)[:k]
            avg = float(np.mean(top))

            if avg > best_score and avg >= self.MATCH_THRESHOLD:
                best_score = avg
                best_name  = name

        return best_name, round(best_score, 4)

    def _estimate_pose(self, landmarks, shape) -> tuple:
        h, w = shape[:2]
        lm   = {lm[0]: (lm[1], lm[2]) for lm in landmarks}

        REQUIRED = [1, 152, 33, 263, 61, 291]
        for idx in REQUIRED:
            if idx not in lm:
                return 0.0, 0.0, 0.0

        image_points = np.array([
            lm[1], lm[152], lm[33],
            lm[263], lm[61], lm[291],
        ], dtype=np.float64)

        model_points = np.array([
            ( 0.0,    0.0,    0.0),
            ( 0.0,  -63.6,  -12.5),
            (-43.3,  32.7,  -26.0),
            ( 43.3,  32.7,  -26.0),
            (-28.0, -28.0,  -24.0),
            ( 28.0, -28.0,  -24.0),
        ], dtype=np.float64)

        focal      = w * 1.2
        cam_matrix = np.array(
            [[focal, 0, w/2],
             [0, focal, h/2],
             [0,     0,   1]], dtype=np.float64
        )

        ok, rvec, _ = cv2.solvePnP(
            model_points, image_points,
            cam_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy       = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)

        if sy >= 1e-6:
            pitch = np.arctan2( rmat[2,1], rmat[2,2])
            yaw   = np.arctan2(-rmat[2,0], sy)
            roll  = np.arctan2( rmat[1,0], rmat[0,0])
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1])
            yaw   = np.arctan2(-rmat[2,0], sy)
            roll  = 0.0

        yaw, pitch, roll = (np.degrees(v) for v in (yaw, pitch, roll))
        yaw  = -yaw
        roll = -roll

        if not hasattr(self, "prev_angles"):
            self.prev_angles = (yaw, pitch, roll)

        def smooth(prev, curr):
            diff  = abs(curr - prev)
            alpha = (0.9 if diff < 1 else
                     0.8 if diff < 5 else
                     0.6 if diff < 15 else 0.3)
            return alpha * prev + (1 - alpha) * curr

        yaw   = smooth(self.prev_angles[0], yaw)
        pitch = smooth(self.prev_angles[1], pitch)
        roll  = smooth(self.prev_angles[2], roll)

        if not hasattr(self, "angle_buffer"):
            self.angle_buffer = []
        self.angle_buffer.append((yaw, pitch, roll))
        if len(self.angle_buffer) > 5:
            self.angle_buffer.pop(0)

        yaw   = np.mean([x[0] for x in self.angle_buffer])
        pitch = np.mean([x[1] for x in self.angle_buffer])
        roll  = np.mean([x[2] for x in self.angle_buffer])

        yaw -= self.YAW_OFFSET
        self.prev_angles = (yaw, pitch, roll)

        return round(yaw, 1), round(pitch, 1), round(roll, 1)

    def _pose_to_tick(self, yaw: float, pitch: float) -> tuple:
        norm_yaw   = np.clip(yaw   / self.YAW_MAX,   -1.0, 1.0)
        norm_pitch = np.clip(pitch / self.PITCH_MAX, -1.0, 1.0)
        magnitude  = np.sqrt(norm_yaw**2 + norm_pitch**2)
        angle_deg  = np.degrees(np.arctan2(norm_pitch, norm_yaw)) + 90
        if angle_deg < 0:
            angle_deg += 360
        tick_idx = int((angle_deg / 360) * self.TICK_COUNT) % self.TICK_COUNT
        return tick_idx, round(float(magnitude), 3)

    def _is_face_in_circle(self, landmarks, cx, cy) -> bool:
        r          = self.CIRCLE_RADIUS
        KEY_POINTS = [4, 152, 10, 234, 454, 33, 263]
        lm_dict    = {lm[0]: (lm[1], lm[2]) for lm in landmarks}
        for idx in KEY_POINTS:
            if idx not in lm_dict:
                return False
            px, py = lm_dict[idx]
            if (px - cx)**2 + (py - cy)**2 > (r * 0.85)**2:
                return False
        return True

    def _build_frame(self, frame, cx, cy) -> np.ndarray:
        h, w     = frame.shape[:2]
        r        = self.CIRCLE_RADIUS
        blurred  = cv2.GaussianBlur(frame, (55, 55), 0)
        darkened = (blurred * 0.3).astype(np.uint8)
        display  = darkened.copy()
        mask     = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        display[mask == 255] = frame[mask == 255]
        return display

    def _draw_ticks(self, frame, cx, cy,
                    visited_angles: set, current_tick: int = -1):
        r, n, gap = self.CIRCLE_RADIUS, self.TICK_COUNT, 8
        FALLOFF   = 6

        for i in range(n):
            angle_rad = np.radians((360 / n) * i - 90)
            cos_a     = np.cos(angle_rad)
            sin_a     = np.sin(angle_rad)

            if i in visited_angles:
                if current_tick >= 0:
                    dist = min(abs(i - current_tick),
                               n - abs(i - current_tick))
                else:
                    dist = FALLOFF + 1

                if dist == 0:
                    brightness, tick_len, thickness = 255, 26, 4
                elif dist <= FALLOFF:
                    factor     = 1.0 - (dist / FALLOFF) ** 1.5
                    brightness = int(120 + 135 * factor)
                    tick_len   = int(14  + 12  * factor)
                    thickness  = 3 if dist <= 2 else 2
                else:
                    brightness, tick_len, thickness = 200, 18, 2

                color = (brightness, brightness, brightness)
            else:
                tick_len, thickness, color = 9, 1, (55, 55, 55)

            x1 = int(cx + (r + gap)            * cos_a)
            y1 = int(cy + (r + gap)            * sin_a)
            x2 = int(cx + (r + gap + tick_len) * cos_a)
            y2 = int(cy + (r + gap + tick_len) * sin_a)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

            if i == current_tick and i in visited_angles:
                cv2.circle(frame, (x2, y2), 3, (255, 255, 255), -1)

    def _draw_ticks_solid(self, frame, cx, cy, brightness, color=None):
        r, n, gap = self.CIRCLE_RADIUS, self.TICK_COUNT, 8
        c = color or (brightness, brightness, brightness)
        for i in range(n):
            angle_rad = np.radians((360 / n) * i - 90)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            x1 = int(cx + (r + gap)      * cos_a)
            y1 = int(cy + (r + gap)      * sin_a)
            x2 = int(cx + (r + gap + 20) * cos_a)
            y2 = int(cy + (r + gap + 20) * sin_a)
            cv2.line(frame, (x1, y1), (x2, y2), c, 2)

    def _draw_text(self, frame, cx, cy, msg: str):
        r, font, fs, th = self.CIRCLE_RADIUS, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        y = cy + r + 55
        for i, line in enumerate(msg.split("\n")):
            (tw, lh), _ = cv2.getTextSize(line, font, fs, th)
            cv2.putText(frame, line,
                        (cx - tw//2, y + i*(lh+14)),
                        font, fs, (255, 255, 255), th)

    def _draw_text_alpha(self, frame, cx, cy, msg: str, alpha: float):
        r, font, fs, th = self.CIRCLE_RADIUS, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        color = tuple(int(255 * alpha) for _ in range(3))
        (tw, _), _ = cv2.getTextSize(msg, font, fs, th)
        cv2.putText(frame, msg,
                    (cx - tw//2, cy + r + 55),
                    font, fs, color, th)

    def _draw_checkmark(self, frame, cx, cy,
                        alpha: float, color=(255, 255, 255)):
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, (cx-40, cy),    (cx-10, cy+30), c, 4)
        cv2.line(frame, (cx-10, cy+30), (cx+45, cy-35), c, 4)

    def _draw_timeout_bar(self, frame, w, elapsed, timeout):
        remaining = max(0.0, 1.0 - elapsed / timeout)
        bar_w     = int(w * remaining)
        color     = (0, 200, 100) if remaining > 0.4 else (0, 80, 255)
        cv2.rectangle(frame, (0, 0), (bar_w, 4), color, -1)

    def _animate_completion(self, cap, last_frame, cx, cy):
        for f in range(20):
            ret, frame = cap.read()
            frame   = cv2.flip(frame, 1) if ret else last_frame.copy()
            display = self._build_frame(frame, cx, cy)
            self._draw_ticks_solid(display, cx, cy,
                                   int(180 + 75 * np.sin(f * 0.4)))
            cv2.imshow("Face ID", display)
            cv2.waitKey(30)

        for f in range(30):
            ret, frame = cap.read()
            frame   = cv2.flip(frame, 1) if ret else last_frame.copy()
            display = self._build_frame(frame, cx, cy)
            alpha   = min(f / 20.0, 1.0)
            self._draw_ticks_solid(display, cx, cy, 255)
            self._draw_checkmark(display, cx, cy, alpha)
            if f > 15:
                self._draw_text_alpha(display, cx, cy,
                                      "Enrollment Complete",
                                      (f - 15) / 15.0)
            cv2.imshow("Face ID", display)
            cv2.waitKey(40)

        cv2.waitKey(1000)

    def _animate_success(self, cap, last_frame, cx, cy, name):
        for f in range(35):
            ret, frame = cap.read()
            frame   = cv2.flip(frame, 1) if ret else last_frame.copy()
            display = self._build_frame(frame, cx, cy)
            alpha   = min(f / 15.0, 1.0)
            self._draw_ticks_solid(display, cx, cy, 255,
                                   color=(0, 220, 100))
            self._draw_checkmark(display, cx, cy, alpha,
                                  color=(0, 255, 100))
            if f > 10:
                self._draw_text_alpha(display, cx, cy,
                                      f"Welcome, {name}!",
                                      (f - 10) / 15.0)
            cv2.imshow("Face ID", display)
            cv2.waitKey(40)
        cv2.waitKey(800)


    def _save_profiles(self):
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._profile_path, "wb") as f:
            pickle.dump(self._profiles, f)

    def _load_profiles(self) -> dict:
        if self._profile_path.exists():
            with open(self._profile_path, "rb") as f:
                return pickle.load(f)
        return {}

    def list_users(self):
        print(f"\n👥 Enrolled: {list(self._profiles.keys()) or 'None'}")

    def remove_user(self, name):
        if name in self._profiles:
            del self._profiles[name]
            self._save_profiles()
            print(f"🗑  Removed '{name}'")
        else:
            print(f"'{name}' not found.")


if __name__ == "__main__":
    import sys
    cmd  = sys.argv[1] if len(sys.argv) > 1 else ""
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    faceid = FaceID()

    if cmd == "enroll":
        faceid.enroll(name or input("Name: "))
    elif cmd == "login":
        faceid.login()
    elif cmd == "list":
        faceid.list_users()
    elif cmd == "remove":
        faceid.remove_user(name or input("Name to remove: "))
    else:
        print("\nUsage:")
        print("  python FaceID.py enroll 'Your Name'")
        print("  python FaceID.py login")
        print("  python FaceID.py list")
        print("  python FaceID.py remove 'Name'")
