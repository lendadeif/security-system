# # face_id.py
# import cv2
# import numpy as np
# import pickle
# import time
# from pathlib import Path
# from FaceMesh import FaceMesh


# class FaceID:

#     TOTAL_CAPTURES = 40
#     MOVE_THRESHOLD = 18
#     INITIAL_HOLD   = 10
#     TICK_COUNT     = 60
#     CIRCLE_RADIUS  = 180

#     def __init__(self):
#         self.mesh          = FaceMesh()
#         self._profile_path = Path("database/face_id_profiles.pkl")
#         self._profiles     = self._load_profiles()


#     def enroll(self, name: str) -> bool:
#         cap            = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         initial_hold   = 0
#         started        = False
#         last_nose      = None
#         captures       = []
#         visited_angles = set()
#         current_tick   = -1

#         print(f"\nEnrolling: {name}")
#         print("  Place your face in the circle, then gently move your head.")
#         print("  Press ESC to cancel.\n")

#         while len(visited_angles) < self.TICK_COUNT:
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             frame  = cv2.flip(frame, 1)
#             h, w   = frame.shape[:2]
#             cx, cy = w // 2, h // 2

#             try:
#                 _, landmarks   = self.mesh.detect(frame, Draw=False)
#                 face_detected  = True
#                 face_in_circle = self._is_face_in_circle(landmarks, cx, cy)
#             except Exception:
#                 face_detected  = False
#                 face_in_circle = False
#                 landmarks      = []

#             nose_pos = None
#             if face_in_circle and landmarks:
#                 lm_dict  = {lm[0]: (lm[1], lm[2]) for lm in landmarks}
#                 nose_pos = lm_dict.get(4, None)

#             if not started:
#                 if face_in_circle:
#                     initial_hold += 1
#                 else:
#                     initial_hold = max(0, initial_hold - 1)

#                 if initial_hold >= self.INITIAL_HOLD:
#                     started   = True
#                     last_nose = nose_pos
#                     print(" Started — gently move your head in a circle!")

#             elif face_in_circle and nose_pos is not None:
#                 current_tick = self._nose_to_tick(nose_pos, cx, cy)

#                 if last_nose is None:
#                     last_nose = nose_pos

#                 dx       = nose_pos[0] - last_nose[0]
#                 dy       = nose_pos[1] - last_nose[1]
#                 distance = np.sqrt(dx**2 + dy**2)

#                 if distance >= self.MOVE_THRESHOLD:
#                     prev_count = len(visited_angles)

#                     for offset in range(-2, 3):
#                         visited_angles.add((current_tick + offset) % self.TICK_COUNT)

#                     if len(visited_angles) > prev_count:
#                         print(f"{len(visited_angles)}/{self.TICK_COUNT} ticks")
#                         cv2.imwrite(f"captures/{name}_{len(visited_angles)}.jpg", frame)

#                     vec = self._landmarks_to_vector(landmarks)
#                     if vec is not None:
#                         captures.append(vec)
#                     last_nose = nose_pos

#             display = self._build_frame(frame, cx, cy)
#             self._draw_ticks(display, cx, cy, visited_angles, current_tick)

#             if not face_detected or not face_in_circle:
#                 msg = "Position your face in the circle"
#             elif not started:
#                 pct = int((initial_hold / self.INITIAL_HOLD) * 100)
#                 msg = f"Hold still...  {pct}%"
#             else:
#                 msg = "Gently move your head to\ncomplete the circle."

#             self._draw_text(display, cx, cy, msg)

#             cv2.imshow("Face ID", display)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 print("\nCancelled.")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return False

#         self._animate_completion(cap, frame, cx, cy, visited_angles)
#         cap.release()
#         cv2.destroyAllWindows()

#         self._profiles[name] = {"name": name, "vectors": captures}
#         self._save_profiles()
#         print(f"\n'{name}' enrolled!")
#         return True


#     def login(self) -> dict:
#         if not self._profiles:
#             print("No enrolled users.")
#             return {"success": False, "name": None, "score": 0.0}

#         cap         = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         start_time  = time.time()
#         TIMEOUT     = 15
#         CONSEC_NEED = 8
#         consecutive = 0
#         scan_angle  = 0    

#         print("\nFace ID — look at the camera...")

#         while True:
#             elapsed = time.time() - start_time
#             if elapsed > TIMEOUT:
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 print("Timed out.")
#                 return {"success": False, "name": None, "score": 0.0}

#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             frame  = cv2.flip(frame, 1)
#             h, w   = frame.shape[:2]
#             cx, cy = w // 2, h // 2

#             try:
#                 _, landmarks   = self.mesh.detect(frame, Draw=False)
#                 face_detected  = True
#                 face_in_circle = self._is_face_in_circle(landmarks, cx, cy)
#             except Exception:
#                 face_detected  = False
#                 face_in_circle = False
#                 landmarks      = []
#                 consecutive    = 0

#             best_name, best_score = "Unknown", 0.0

#             if face_in_circle and landmarks:
#                 vec = self._landmarks_to_vector(landmarks)
#                 if vec is not None:
#                     best_name, best_score = self._match(vec)
#                     consecutive = consecutive + 1 if best_score >= 0.97 else 0

#             if consecutive >= CONSEC_NEED:
#                 display = self._build_frame(frame, cx, cy)
#                 self._draw_ticks(display, cx, cy,
#                                 set(range(self.TICK_COUNT)), -1)
#                 self._animate_success(cap, frame, cx, cy, best_name)
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 print(f"\nWelcome, {best_name}!  score={best_score:.4f}")
#                 return {"success": True, "name": best_name, "score": best_score}

#             scan_visited = set()
#             for offset in range(-4, 5):
#                 scan_visited.add((scan_angle + offset) % self.TICK_COUNT)
#             scan_angle = (scan_angle + 2) % self.TICK_COUNT

#             display = self._build_frame(frame, cx, cy)
#             self._draw_ticks(display, cx, cy, scan_visited, scan_angle)

#             if not face_detected or not face_in_circle:
#                 msg = "Look at the camera"
#             elif best_score > 0.90:
#                 msg = f"Verifying..."
#             else:
#                 msg = "Scanning..."

#             self._draw_text(display, cx, cy, msg)

#             self._draw_timeout_bar(display, w, elapsed, TIMEOUT)

#             cv2.imshow("Face ID", display)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         return {"success": False, "name": None, "score": 0.0}

#     def verify(self, frame: np.ndarray) -> dict:
        
#         if not self._profiles:
#             return {"authorized": False, "name": "Unknown", "score": 0.0}

#         h, w   = frame.shape[:2]
#         cx, cy = w // 2, h // 2

#         try:
#             _, landmarks = self.mesh.detect(frame, Draw=False)
#         except Exception:
#             return {"authorized": False, "name": "Unknown", "score": 0.0}

#         if not landmarks:
#             return {"authorized": False, "name": "Unknown", "score": 0.0}

#         vec = self._landmarks_to_vector(landmarks)
#         if vec is None:
#             return {"authorized": False, "name": "Unknown", "score": 0.0}

#         name, score = self._match(vec)
#         authorized  = score >= 0.97

#         return {
#             "authorized": authorized,
#             "name":       name if authorized else "Unknown",
#             "score":      score
#         }

#     def _animate_completion(self, cap, last_frame, cx, cy, visited_angles):

#         h, w = last_frame.shape[:2]

#         for f in range(20):
#             ret, frame = cap.read()
#             frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
#             display    = self._build_frame(frame, cx, cy)

#             brightness = int(180 + 75 * np.sin(f * 0.4))
#             self._draw_ticks_solid(display, cx, cy, brightness)
#             cv2.imshow("Face ID", display)
#             cv2.waitKey(30)

#         for f in range(30):
#             ret, frame = cap.read()
#             frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
#             display    = self._build_frame(frame, cx, cy)

#             alpha = min(f / 20.0, 1.0)   
#             self._draw_ticks_solid(display, cx, cy, 255)

#             self._draw_checkmark(display, cx, cy, alpha)

#             # Text
#             if f > 15:
#                 text_alpha = (f - 15) / 15.0
#                 self._draw_text_alpha(display, cx, cy,
#                                     "Enrollment Complete", text_alpha)

#             cv2.imshow("Face ID", display)
#             cv2.waitKey(40)

#         cv2.waitKey(1000)

#     def _animate_success(self, cap, last_frame, cx, cy, name):
        
#         h, w = last_frame.shape[:2]

#         for f in range(35):
#             ret, frame = cap.read()
#             frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
#             display    = self._build_frame(frame, cx, cy)

#             alpha = min(f / 15.0, 1.0)

#             self._draw_ticks_solid(display, cx, cy, 255, color=(0, 220, 100))

#             self._draw_checkmark(display, cx, cy, alpha, color=(0, 255, 100))

#             if f > 10:
#                 text_alpha = (f - 10) / 15.0
#                 self._draw_text_alpha(display, cx, cy,
#                                     f"Welcome, {name}!", text_alpha)

#             cv2.imshow("Face ID", display)
#             cv2.waitKey(40)

#         cv2.waitKey(800)


#     def _build_frame(self, frame, cx, cy):
#         h, w = frame.shape[:2]
#         r    = self.CIRCLE_RADIUS

#         blurred  = cv2.GaussianBlur(frame, (55, 55), 0)
#         darkened = (blurred * 0.3).astype(np.uint8)
#         display  = darkened.copy()

#         mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.circle(mask, (cx, cy), r, 255, -1)
#         display[mask == 255] = frame[mask == 255]

#         return display

#     def _draw_ticks(self, frame, cx, cy, visited_angles: set, current_tick: int = -1):
#         r       = self.CIRCLE_RADIUS
#         n       = self.TICK_COUNT
#         gap     = 8
#         FALLOFF = 6

#         for i in range(n):
#             angle_deg = (360 / n) * i - 90
#             angle_rad = np.radians(angle_deg)
#             cos_a     = np.cos(angle_rad)
#             sin_a     = np.sin(angle_rad)

#             if i in visited_angles:
#                 if current_tick >= 0:
#                     dist = min(
#                         abs(i - current_tick),
#                         n - abs(i - current_tick)
#                     )
#                 else:
#                     dist = FALLOFF + 1

#                 if dist == 0:
#                     brightness = 255
#                     tick_len   = 26
#                     thickness  = 4
#                 elif dist <= FALLOFF:
#                     factor     = 1.0 - (dist / FALLOFF) ** 1.5
#                     brightness = int(120 + 135 * factor)
#                     tick_len   = int(14 + 12 * factor)
#                     thickness  = 3 if dist <= 2 else 2
#                 else:
#                     brightness = 200
#                     tick_len   = 18
#                     thickness  = 2

#                 color = (brightness, brightness, brightness)

#             else:
#                 brightness = 55
#                 tick_len   = 9
#                 thickness  = 1
#                 color      = (brightness, brightness, brightness)

#             x1 = int(cx + (r + gap) * cos_a)
#             y1 = int(cy + (r + gap) * sin_a)
#             x2 = int(cx + (r + gap + tick_len) * cos_a)
#             y2 = int(cy + (r + gap + tick_len) * sin_a)
#             cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

#             # Glow dot at peak
#             if i == current_tick and i in visited_angles:
#                 cv2.circle(frame, (x2, y2), 3, (255, 255, 255), -1)

#     def _draw_ticks_solid(self, frame, cx, cy, brightness, color=None):
#         r   = self.CIRCLE_RADIUS
#         n   = self.TICK_COUNT
#         gap = 8
#         c   = color if color else (brightness, brightness, brightness)

#         for i in range(n):
#             angle_rad = np.radians((360 / n) * i - 90)
#             cos_a     = np.cos(angle_rad)
#             sin_a     = np.sin(angle_rad)
#             x1 = int(cx + (r + gap) * cos_a)
#             y1 = int(cy + (r + gap) * sin_a)
#             x2 = int(cx + (r + gap + 20) * cos_a)
#             y2 = int(cy + (r + gap + 20) * sin_a)
#             cv2.line(frame, (x1, y1), (x2, y2), c, 2)

#     def _draw_checkmark(self, frame, cx, cy, alpha, color=(255, 255, 255)):
#         thickness = 4
#         c = tuple(int(v * alpha) for v in color)

#         p1 = (cx - 40, cy)
#         p2 = (cx - 10, cy + 30)
#         p3 = (cx + 45, cy - 35)

#         cv2.line(frame, p1, p2, c, thickness)
#         cv2.line(frame, p2, p3, c, thickness)

#     def _draw_text(self, frame, cx, cy, msg: str):
#         r         = self.CIRCLE_RADIUS
#         font      = cv2.FONT_HERSHEY_SIMPLEX
#         font_size = 0.75
#         thickness = 2
#         y_start   = cy + r + 55

#         for i, line in enumerate(msg.split("\n")):
#             (tw, th), _ = cv2.getTextSize(line, font, font_size, thickness)
#             tx = cx - tw // 2
#             ty = y_start + i * (th + 14)
#             cv2.putText(frame, line, (tx, ty), font,
#                         font_size, (255, 255, 255), thickness)

#     def _draw_text_alpha(self, frame, cx, cy, msg: str, alpha: float):

#         r         = self.CIRCLE_RADIUS
#         font      = cv2.FONT_HERSHEY_SIMPLEX
#         font_size = 0.85
#         thickness = 2
#         color     = tuple(int(255 * alpha) for _ in range(3))
#         y         = cy + r + 55

#         (tw, _), _ = cv2.getTextSize(msg, font, font_size, thickness)
#         cv2.putText(frame, msg, (cx - tw // 2, y),
#                     font, font_size, color, thickness)

#     def _draw_timeout_bar(self, frame, w, elapsed, timeout):
#         remaining = max(0.0, 1.0 - elapsed / timeout)
#         bar_w     = int(w * remaining)
#         color     = (0, 200, 100) if remaining > 0.4 else (0, 80, 255)
#         cv2.rectangle(frame, (0, 0), (bar_w, 4), color, -1)


#     def _nose_to_tick(self, nose_pos, cx, cy) -> int:
#         nx, ny    = nose_pos
#         angle_deg = np.degrees(np.arctan2(ny - cy, nx - cx)) + 90
#         if angle_deg < 0:
#             angle_deg += 360
#         return int((angle_deg / 360) * self.TICK_COUNT) % self.TICK_COUNT

#     def _is_face_in_circle(self, landmarks, cx, cy) -> bool:
#         r          = self.CIRCLE_RADIUS
#         KEY_POINTS = [4, 152, 10, 234, 454, 33, 263]
#         lm_dict    = {lm[0]: (lm[1], lm[2]) for lm in landmarks}
#         for idx in KEY_POINTS:
#             if idx not in lm_dict:
#                 return False
#             px, py = lm_dict[idx]
#             if (px - cx)**2 + (py - cy)**2 > (r * 0.85)**2:
#                 return False
#         return True

#     def _landmarks_to_vector(self, landmarks):
#         if not landmarks:
#             return None
#         lm_dict = {lm[0]: np.array([lm[1], lm[2]], dtype=np.float32)
#                 for lm in landmarks}
#         if 4 not in lm_dict or 33 not in lm_dict or 263 not in lm_dict:
#             return None
#         origin   = lm_dict[4]
#         pts      = np.array(
#             [lm_dict.get(i, origin) - origin for i in range(len(landmarks))],
#             dtype=np.float32
#         )
#         eye_dist = np.linalg.norm(lm_dict[33] - lm_dict[263])
#         if eye_dist < 1e-6:
#             return None
#         return (pts / eye_dist).flatten()

#     def _cosine_similarity(self, a, b) -> float:
#         denom = np.linalg.norm(a) * np.linalg.norm(b)
#         if denom < 1e-9:
#             return 0.0
#         return float(np.dot(a, b) / denom)

#     def _match(self, vec) -> tuple:
#         best_name, best_score = "Unknown", 0.0
#         for name, profile in self._profiles.items():
#             scores = [self._cosine_similarity(vec, v)
#                     for v in profile["vectors"]]
#             top3   = sorted(scores, reverse=True)[:3]
#             avg    = float(np.mean(top3))
#             if avg > best_score:
#                 best_score = avg
#                 best_name  = name
#         return best_name, round(best_score, 4)


#     def _save_profiles(self):
#         self._profile_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(self._profile_path, "wb") as f:
#             pickle.dump(self._profiles, f)

#     def _load_profiles(self) -> dict:
#         if self._profile_path.exists():
#             with open(self._profile_path, "rb") as f:
#                 return pickle.load(f)
#         return {}

#     def list_users(self):
#         users = list(self._profiles.keys())
#         print(f"\nEnrolled: {users or 'None'}")
#         return users

#     def remove_user(self, name):
#         if name in self._profiles:
#             del self._profiles[name]
#             self._save_profiles()
#             print(f"Removed '{name}'")
#         else:
#             print(f"'{name}' not found.")


# if __name__ == "__main__":
#     import sys
#     cmd  = sys.argv[1] if len(sys.argv) > 1 else ""
#     name = sys.argv[2] if len(sys.argv) > 2 else ""

#     faceid = FaceID()

#     if cmd == "enroll":
#         if not name:
#             name = input("Name: ")
#         faceid.enroll(name)

#     elif cmd == "login":
#         faceid.login()

#     elif cmd == "list":
#         faceid.list_users()

#     elif cmd == "remove":
#         if not name:
#             name = input("Name to remove: ")
#         faceid.remove_user(name)

#     else:
#         print("\nUsage:")
#         print("  python face_id.py enroll 'Your Name'")
#         print("  python face_id.py login")
#         print("  python face_id.py list")
#         print("  python face_id.py remove 'Name'")


# face_id.py
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from FaceMesh import FaceMesh


class FaceID:

    TICK_COUNT     = 60     
    CIRCLE_RADIUS  = 180    
    INITIAL_HOLD   = 25     

    # Pose settings
    MIN_MAGNITUDE  = 0.25   
                            
    YAW_MAX        = 35.0   
    PITCH_MAX      = 25.0   

    YAW_OFFSET     = 14.0

    TICK_SPREAD    = 2

    def __init__(self):
        self.mesh          = FaceMesh()
        self._profile_path = Path("database/face_id_profiles.pkl")
        self._profiles     = self._load_profiles()


    def enroll(self, name: str) -> bool:
        cap            = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        initial_hold   = 0
        started        = False
        captures       = []            
        visited_angles = set()        
        current_tick   = -1           
        current_mag    = 0.0          

        print(f"\nEnrolling: {name}")
        print("  Place your face in the circle, then gently move your head.")
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
                yaw, pitch, roll = self._estimate_pose(landmarks, frame.shape)
                current_tick, current_mag = self._pose_to_tick(yaw, pitch)

            if not started:
                if face_in_circle:
                    initial_hold += 1
                else:
                    initial_hold = max(0, initial_hold - 1)

                if face_in_circle and landmarks and initial_hold == self.INITIAL_HOLD // 2:
                    vec = self._landmarks_to_vector(landmarks)
                    if vec is not None:
                        captures.append(vec)

                if initial_hold >= self.INITIAL_HOLD:
                    started = True
                    print("  Started — gently move your head in a circle!")

            elif face_in_circle and landmarks:

                if current_mag >= self.MIN_MAGNITUDE:
                    prev_count = len(visited_angles)

                    # Light up current tick + neighbors
                    for offset in range(-self.TICK_SPREAD, self.TICK_SPREAD + 1):
                        visited_angles.add(
                            (current_tick + offset) % self.TICK_COUNT
                        )

                    if len(visited_angles) > prev_count:
                        vec = self._landmarks_to_vector(landmarks)
                        if vec is not None:
                            captures.append(vec)
                            print(
                                f" {len(visited_angles)}/{self.TICK_COUNT} "
                                f"ticks  yaw={yaw:+.1f}  pitch={pitch:+.1f}"
                            )

            display = self._build_frame(frame, cx, cy)
            self._draw_ticks(display, cx, cy, visited_angles, current_tick
                            if started and current_mag >= self.MIN_MAGNITUDE
                            else -1)

            if not face_detected or not face_in_circle:
                msg = "Position your face in the circle"
            elif not started:
                pct = int((initial_hold / self.INITIAL_HOLD) * 100)
                msg = f"Hold still...  {pct}%"
            else:
                msg = "Gently move your head to\ncomplete the circle."

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

        self._profiles[name] = {"name": name, "vectors": captures}
        self._save_profiles()
        print(f"\n'{name}' enrolled with {len(captures)} captures!")
        return True


    def login(self) -> dict:
        if not self._profiles:
            print("No enrolled users.")
            return {"success": False, "name": None, "score": 0.0}

        cap         = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        start_time  = time.time()
        TIMEOUT     = 15
        CONSEC_NEED = 8
        consecutive = 0
        scan_angle  = 0

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

            best_name, best_score = "Unknown", 0.0

            if face_in_circle and landmarks:
                vec = self._landmarks_to_vector(landmarks)
                if vec is not None:
                    best_name, best_score = self._match(vec)
                    consecutive = consecutive + 1 if best_score >= 0.97 else 0

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
            elif best_score > 0.90:
                msg = "Verifying..."
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


    def verify(self, frame: np.ndarray) -> dict:

        if not self._profiles:
            return {"authorized": False, "name": "Unknown", "score": 0.0}
        try:
            _, landmarks = self.mesh.detect(frame, Draw=False)
        except Exception:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        if not landmarks:
            return {"authorized": False, "name": "Unknown", "score": 0.0}

        vec = self._landmarks_to_vector(landmarks)
        if vec is None:
            return {"authorized": False, "name": "Unknown", "score": 0.0}
        name, score = self._match(vec)
        authorized  = score >= 0.6

        return {
            "authorized": authorized,
            "name":       name if authorized else "Unknown",
            "score":      score
        }


    def _estimate_pose(self, landmarks, shape) -> tuple:

        h, w = shape[:2]
        lm   = {lm[0]: (lm[1], lm[2]) for lm in landmarks}

        REQUIRED = [4, 152, 33, 263, 61, 291]
        for idx in REQUIRED:
            if idx not in lm:
                return 0.0, 0.0, 0.0

        image_points = np.array([
            lm[4],  lm[152],
            lm[33], lm[263],
            lm[61], lm[291],
        ], dtype=np.float64)

        if image_points.shape != (6, 2):
            return 0.0, 0.0, 0.0

        model_points = np.array([
            ( 0.0,    0.0,    0.0),
            ( 0.0,  -63.6,  -12.5),
            (-43.3,  32.7,  -26.0),
            ( 43.3,  32.7,  -26.0),
            (-28.9, -28.9,  -24.1),
            ( 28.9, -28.9,  -24.1),
        ], dtype=np.float64)

        focal      = float(w)
        cam_matrix = np.array(
            [[focal, 0, w / 2],
             [0, focal, h / 2],
             [0,     0,     1]], dtype=np.float64
        )

        ok, rvec, _ = cv2.solvePnP(
            model_points, image_points,
            cam_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _    = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw, pitch, roll = angles[1], angles[0], angles[2]

        # Fix gimbal lock
        for val in [yaw, pitch, roll]:
            pass
        if pitch > 90:    pitch -= 180
        elif pitch < -90: pitch += 180
        if yaw   > 90:    yaw   -= 180
        elif yaw   < -90: yaw   += 180
        if roll  > 90:    roll  -= 180
        elif roll  < -90: roll  += 180

        yaw -= self.YAW_OFFSET

        return round(yaw, 1), round(pitch, 1), round(roll, 1)

    def _pose_to_tick(self, yaw: float, pitch: float) -> tuple:
        

        norm_yaw   = np.clip(yaw   / self.YAW_MAX,   -1.0, 1.0)
        norm_pitch = np.clip(pitch / self.PITCH_MAX, -1.0, 1.0)

        magnitude = np.sqrt(norm_yaw**2 + norm_pitch**2)

        angle_deg = np.degrees(np.arctan2(norm_pitch, norm_yaw)) + 90
        if angle_deg < 0:
            angle_deg += 360

        tick_idx = int((angle_deg / 360) * self.TICK_COUNT) % self.TICK_COUNT

        return tick_idx, round(float(magnitude), 3)


    def _landmarks_to_vector(self, landmarks) -> np.ndarray | None:
        if not landmarks:
            return None

        lm_dict = {lm[0]: np.array([lm[1], lm[2]], dtype=np.float32)
                for lm in landmarks}

        if 4 not in lm_dict or 33 not in lm_dict or 263 not in lm_dict:
            return None

        origin   = lm_dict[4]
        pts      = np.array(
            [lm_dict.get(i, origin) - origin for i in range(len(landmarks))],
            dtype=np.float32
        )
        eye_dist = np.linalg.norm(lm_dict[33] - lm_dict[263])
        if eye_dist < 1e-6:
            return None

        return (pts / eye_dist).flatten()

    def _cosine_similarity(self, a, b) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _match(self, vec) -> tuple:
        """Compare against all enrolled profiles. Returns (name, score)."""
        best_name, best_score = "Unknown", 0.0
        for name, profile in self._profiles.items():
            scores = [self._cosine_similarity(vec, v)
                    for v in profile["vectors"]]
            top3   = sorted(scores, reverse=True)[:3]
            avg    = float(np.mean(top3))
            if avg > best_score:
                best_score = avg
                best_name  = name
        return best_name, round(best_score, 4)


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
        r       = self.CIRCLE_RADIUS
        n       = self.TICK_COUNT
        gap     = 8
        FALLOFF = 6

        for i in range(n):
            angle_rad = np.radians((360 / n) * i - 90)
            cos_a     = np.cos(angle_rad)
            sin_a     = np.sin(angle_rad)

            if i in visited_angles:
                if current_tick >= 0:
                    dist = min(
                        abs(i - current_tick),
                        n - abs(i - current_tick)
                    )
                else:
                    dist = FALLOFF + 1

                if dist == 0:
                    brightness = 255
                    tick_len   = 26
                    thickness  = 4
                elif dist <= FALLOFF:
                    factor     = 1.0 - (dist / FALLOFF) ** 1.5
                    brightness = int(120 + 135 * factor)
                    tick_len   = int(14 + 12 * factor)
                    thickness  = 3 if dist <= 2 else 2
                else:
                    brightness = 200
                    tick_len   = 18
                    thickness  = 2

                color = (brightness, brightness, brightness)
            else:
                tick_len  = 9
                thickness = 1
                color     = (55, 55, 55)

            x1 = int(cx + (r + gap) * cos_a)
            y1 = int(cy + (r + gap) * sin_a)
            x2 = int(cx + (r + gap + tick_len) * cos_a)
            y2 = int(cy + (r + gap + tick_len) * sin_a)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

            # Glow dot at peak tick tip
            if i == current_tick and i in visited_angles:
                cv2.circle(frame, (x2, y2), 3, (255, 255, 255), -1)

    def _draw_ticks_solid(self, frame, cx, cy,
                        brightness: int, color=None):
        r   = self.CIRCLE_RADIUS
        n   = self.TICK_COUNT
        gap = 8
        c   = color or (brightness, brightness, brightness)
        for i in range(n):
            angle_rad = np.radians((360 / n) * i - 90)
            cos_a     = np.cos(angle_rad)
            sin_a     = np.sin(angle_rad)
            x1 = int(cx + (r + gap) * cos_a)
            y1 = int(cy + (r + gap) * sin_a)
            x2 = int(cx + (r + gap + 20) * cos_a)
            y2 = int(cy + (r + gap + 20) * sin_a)
            cv2.line(frame, (x1, y1), (x2, y2), c, 2)

    def _draw_text(self, frame, cx, cy, msg: str):
        r         = self.CIRCLE_RADIUS
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.75
        thickness = 2
        y_start   = cy + r + 55
        for i, line in enumerate(msg.split("\n")):
            (tw, th), _ = cv2.getTextSize(line, font, font_size, thickness)
            tx = cx - tw // 2
            ty = y_start + i * (th + 14)
            cv2.putText(frame, line, (tx, ty), font,
                        font_size, (255, 255, 255), thickness)

    def _draw_text_alpha(self, frame, cx, cy, msg: str, alpha: float):
        r         = self.CIRCLE_RADIUS
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.85
        thickness = 2
        color     = tuple(int(255 * alpha) for _ in range(3))
        (tw, _), _ = cv2.getTextSize(msg, font, font_size, thickness)
        cv2.putText(frame, msg,
                    (cx - tw // 2, cy + r + 55),
                    font, font_size, color, thickness)

    def _draw_checkmark(self, frame, cx, cy,
                        alpha: float, color=(255, 255, 255)):
        c  = tuple(int(v * alpha) for v in color)
        p1 = (cx - 40, cy)
        p2 = (cx - 10, cy + 30)
        p3 = (cx + 45, cy - 35)
        cv2.line(frame, p1, p2, c, 4)
        cv2.line(frame, p2, p3, c, 4)

    def _draw_timeout_bar(self, frame, w, elapsed, timeout):
        remaining = max(0.0, 1.0 - elapsed / timeout)
        bar_w     = int(w * remaining)
        color     = (0, 200, 100) if remaining > 0.4 else (0, 80, 255)
        cv2.rectangle(frame, (0, 0), (bar_w, 4), color, -1)


    def _animate_completion(self, cap, last_frame, cx, cy):
        for f in range(20):
            ret, frame = cap.read()
            frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
            display    = self._build_frame(frame, cx, cy)
            brightness = int(180 + 75 * np.sin(f * 0.4))
            self._draw_ticks_solid(display, cx, cy, brightness)
            cv2.imshow("Face ID", display)
            cv2.waitKey(30)

        for f in range(30):
            ret, frame = cap.read()
            frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
            display    = self._build_frame(frame, cx, cy)
            alpha      = min(f / 20.0, 1.0)
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
            frame      = cv2.flip(frame, 1) if ret else last_frame.copy()
            display    = self._build_frame(frame, cx, cy)
            alpha      = min(f / 15.0, 1.0)
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
        users = list(self._profiles.keys())
        print(f"\nEnrolled: {users or 'None'}")
        return users

    def remove_user(self, name):
        if name in self._profiles:
            del self._profiles[name]
            self._save_profiles()
            print(f"Removed '{name}'")
        else:
            print(f"'{name}' not found.")


if __name__ == "__main__":
    import sys
    cmd  = sys.argv[1] if len(sys.argv) > 1 else ""
    name = sys.argv[2] if len(sys.argv) > 2 else ""

    faceid = FaceID()

    if cmd == "enroll":
        if not name:
            name = input("Name: ")
        faceid.enroll(name)

    elif cmd == "login":
        faceid.login()

    elif cmd == "list":
        faceid.list_users()

    elif cmd == "remove":
        if not name:
            name = input("Name to remove: ")
        faceid.remove_user(name)

    else:
        print("\nUsage:")
        print("  python FaceID.py enroll 'Your Name'")
        print("  python FaceID.py login")
        print("  python FaceID.py list")
        print("  python FaceID.py remove 'Name'")