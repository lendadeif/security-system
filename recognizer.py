import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from insightface.app import FaceAnalysis
from loguru import logger
from config import (
    FACES_DIR, DB_PATH,
    FACE_TOLERANCE,
    FACE_RECOGNITION_INTERVAL
)

class FaceRecognizer:
    def __init__(self):
        self._app=None
        self._authorized={}
        self._frame_counter=0
        self._last_results={}
        self._inference_times=[]
    
    def load(self):
        logger.info("Loading InsightFace (ArcFace) model...")

        self._app = FaceAnalysis(
            name="buffalo_l",       #
            allowed_modules=["detection", "recognition"]
        )
        self._app.prepare(
            ctx_id=-1,              
            det_size=(640, 640)     )

        logger.info("InsightFace loaded.")
        self._load_authorized_db()
    
    def process(self,frame,detections):
        self._frame_counter+=1
        run_recognition=(self._frame_counter%FACE_RECOGNITION_INTERVAL==0)
        for i ,det in enumerate(detections):
            if not run_recognition and i in self._last_results:

                cached=self._last_results[i]
                det.update(cached)
                continue
            x1,y1,x2,y2=det["bbox"]
            pad=20
            h,w=frame.shape[:2]
            crop = frame[
                max(0, y1 - pad): min(h, y2 + pad),
                max(0, x1 - pad): min(w, x2 + pad)
            ]
            if crop.size == 0:
                det.update({"authorized": None, "name": "No crop", "match_score": 0.0})
                continue
            start = time.perf_counter()
            result = self._identify(crop)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._inference_times.append(elapsed_ms)

            det.update(result)

            if result["authorized"] is True:
                det["label"] = f"{result['name']}  {result['match_score']:.0%}"
            elif result["authorized"] is False:
                det["label"] = f"UNAUTHORIZED  {result['match_score']:.0%}"
            else:
                det["label"] = "Identifying..."

            self._last_results[i] = result

            logger.debug(
                f"Person {i+1}: {result['name']} | "
                f"authorized={result['authorized']} | "
                f"score={result['match_score']:.2f} | "
                f"{elapsed_ms:.1f}ms"
            )
        return detections
    def enroll(self,name,image_source):
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
            if img is None:
                logger.error(f"Cannot load image: {image_source}")
                return False
        else:
            img = image_source
        
        faces = self._app.get(img)

        if not faces:
            logger.warning(f"No face found in enrollment image for '{name}'")
            return False

        if len(faces) > 1:
            logger.warning(
                f"{len(faces)} faces found — using the largest one for '{name}'"
            )
        
        face       = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding  = face.normed_embedding

        if name not in self._authorized:
            self._authorized[name] = []
        self._authorized[name].append(embedding)

        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        img_count  = len(list(person_dir.glob("*.jpg")))
        img_path   = person_dir / f"{img_count + 1}.jpg"
        cv2.imwrite(str(img_path), img)

        self._save_authorized_db()
        logger.info(f"Enrolled '{name}' — total embeddings: {len(self._authorized[name])}")
        return True
    
    def remove(self,name):
        if name not in self._authorized:
            logger.warning(f"'{name}' not found in authorized database.")
            return False

        del self._authorized[name]
        self._save_authorized_db()
        logger.info(f"Removed '{name}' from authorized database.")
        return True
    def list_authorized(self):
        return list(self._authorized.keys())
    
    def get_avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return round(sum(self._inference_times) / len(self._inference_times), 1)
    
    def _identify(self,crop):
        empty = {"authorized": None, "name": "No face", "match_score": 0.0}

        if self._app is None:
            return empty
        
        faces = self._app.get(crop)

        if not faces:
            return empty
        face      = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding

        if not self._authorized:
            
            return {"authorized": False, "name": "Unknown", "match_score": 0.0}
        
        best_name  = "Unknown"
        best_score = -1.0

        for name, embeddings in self._authorized.items():
            for auth_emb in embeddings:
                # Cosine similarity (higher = more similar)
                score = float(np.dot(embedding, auth_emb))
                if score > best_score:
                    best_score = score
                    best_name  = name
        
        threshold  = 1.0 - FACE_TOLERANCE
        authorized = best_score >= threshold

        return {
            "authorized":  authorized,
            "name":        best_name if authorized else "Unknown",
            "match_score": round(best_score, 3)
        }
    
    def _save_authorized_db(self):
        db_file = DB_PATH.parent / "embeddings.pkl"
        with open(db_file, "wb") as f:
            pickle.dump(self._authorized, f)
        logger.debug(f"Embeddings saved → {db_file}")

    def _load_authorized_db(self):
        db_file = DB_PATH.parent / "embeddings.pkl"
        if db_file.exists():
            with open(db_file, "rb") as f:
                self._authorized = pickle.load(f)
            names = list(self._authorized.keys())
            logger.info(f"Loaded {len(names)} authorized person(s): {names}")
        else:
            logger.info("No existing embeddings database found — starting fresh.")
            self._authorized = {}
            
    