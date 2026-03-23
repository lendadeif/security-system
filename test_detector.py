# test_detector.py
import cv2
from camera import Camera
from detector import PersonDetector
from loguru import logger


def main():
    logger.info("Starting detector test — press Q to quit, S to save snapshot")

    detector = PersonDetector()
    detector.load()

    with Camera() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # ── Detect people ──────────────────────────────
            detections = detector.detect(frame)

            # ── Draw boxes + stats ─────────────────────────
            annotated = detector.draw(frame, detections)
            detector.draw_stats(annotated, fps=cam.get_fps())

            # ── Person count banner ────────────────────────
            count = len(detections)
            banner_color = (0, 200, 0) if count == 0 else (0, 140, 255)
            cv2.putText(
                annotated,
                f"  {count} person(s) detected  |  Q = quit  |  S = snapshot",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                banner_color, 2
            )

            cv2.imshow("Security System — Detector Test", annotated)

            key = cv2.waitKey(1) & 0xFF

            # Q → quit
            if key == ord('q'):
                logger.info("Test complete!")
                break

            # S → save snapshot
            if key == ord('s'):
                path = f"captures/snapshot_{int(__import__('time').time())}.jpg"
                cv2.imwrite(path, annotated)
                logger.info(f"Snapshot saved → {path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()