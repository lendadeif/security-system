from tabnanny import verbose

from torch import classes

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE,
    DETECTION_INTERVAL, MODELS
)

class PersonDetector:
    PERSON_CLASS_ID=0
    def __init__(self) :
        self._model=None
        self._last_detections=[]
        self._frame_counter=0
        self._detection_times=[]
    
    def load(self):
        model_path=MODELS/YOLO_MODEL
        if not model_path.exists():
            logger.info(f"Downloading {YOLO_MODEL}  (first run only)...")
        else:
            logger.info(f"Loading model from {model_path}")
        
        self._model=YOLO(str(YOLO_MODEL))
        logger.info("YOLOv8-nano loaded — person detector ready.")
    
    def detect(self,frame):
        self._frame_counter+=1
        if self._frame_counter % DETECTION_INTERVAL != 0:
            return self._last_detections
        
        start=time.perf_counter()
        results=self._model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=YOLO_CONFIDENCE,
            verbose=False
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._detection_times.append(elapsed_ms)

        detections = self._parse_results(results, frame.shape)
        self._last_detections = detections

        logger.debug(
            f"Detection: {len(detections)} person(s) found "
            f"in {elapsed_ms:.1f}ms"
        )

        return detections
    
    def get_avg_detection_ms(self):
        if not self._detection_times:
            return 0.0
        return round(sum(self._detection_times) / len(self._detection_times), 1)
    
    def draw(self,frame,detections):
        annotated = frame.copy()
        for i , det in enumerate(detections):
            x1,y1,x2,y2=det["bbox"]
            conf=det["confidence"]
            is_authorized=det.get("authorized",None)

            if is_authorized is None:
                color=(255,165,0)
            elif is_authorized:
                color=(0,255,0)
            else:
                color=(0,0,255)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            
            label      = det.get("label", f"Person {i+1}  {conf:.0%}")
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 10),
                (x1 + tw + 8, y1),
                color, -1   # Filled background
            )

            
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255),   # White text
                2
            )

            # Center dot
            cx, cy = det["center"]
            cv2.circle(annotated, (cx, cy), 4, color, -1)
        return annotated
    
    def draw_stats(self,frame,fps):
        h,w=frame.shape[:2]
        count=len(self._last_detections)
        avg_ms=self.get_avg_detection_ms()

        lines=[
            f"FPS:{fps}",
            f"Persons: {count}",
            f"Detect ms: {avg_ms}"
        ]
        panel_w, panel_h = 220, len(lines) * 28 + 16
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (w - panel_w - 10, 10),
            (w - 10, panel_h + 10),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for idx, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (w - panel_w, 36 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 180), 2
            )

        return frame
    def _parse_results(self,results,frame_shape):
        detections=[]
        h,w=frame_shape[:2]
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2=box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                conf = float(box.conf[0])

                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": conf,
                    "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                    "area":       (x2 - x1) * (y2 - y1),
                    "label":      f"Person  {conf:.0%}",
                    "authorized": None   # Will be filled by recognizer in Step 4
                })
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections
    
