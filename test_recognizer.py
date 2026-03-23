# test_recognizer.py
import cv2
from camera import Camera
from detector import PersonDetector
from recognizer import FaceRecognizer
from loguru import logger


def main():
    logger.info("Starting full detection + recognition test — press Q to quit")

    detector   = PersonDetector()
    recognizer = FaceRecognizer()

    detector.load()
    recognizer.load()

    authorized = recognizer.list_authorized()
    logger.info(f"Authorized persons: {authorized or 'None enrolled yet'}")

    with Camera() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            detections = detector.detect(frame)

            detections = recognizer.process(frame, detections)


            annotated = detector.draw(frame, detections)
            detector.draw_stats(annotated, fps=cam.get_fps())

            unauth = sum(
                1 for d in detections
                if d.get("authorized") is False
            )
            color = (0, 0, 255) if unauth > 0 else (0, 200, 0)
            status = (
                f"{unauth} UNAUTHORIZED" if unauth > 0
                else "✅ All clear"
            )
            cv2.putText(
                annotated, f"  {status}  |  Q = quit",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            cv2.imshow("Security System — Recognition Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()