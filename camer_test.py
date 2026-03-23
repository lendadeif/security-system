# test_camera.py
import cv2
from camera import Camera
from loguru import logger

def main():
    logger.info("Starting camera test — press Q to quit")

    with Camera() as cam:
        while True:
            frame = cam.read()

            if frame is None:
                continue

            
            fps = cam.get_fps()
            cv2.putText(
                frame,
                f"FPS: {fps}  |  Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Security System — Camera Test", frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Test complete!")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()