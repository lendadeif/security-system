import cv2
import argparse
from recognizer import FaceRecognizer
from loguru import logger

def enroll_from_image(recognizer, name, image_path):
    success = recognizer.enroll(name, image_path)
    if success:
        print(f"\n'{name}' enrolled successfully!")
    else:
        print(f"\nFailed — make sure the image has a clear, visible face.")


def enroll_from_webcam(recognizer, name):
    print(f"\nWebcam enrollment for '{name}'")
    print("Position your face clearly in the frame, then press SPACE to capture.")
    print("Press Q to cancel.\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        if not ret:
            continue

        cv2.putText(
            frame,
            "SPACE = capture  |  Q = cancel",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow(f"Enrolling: {name}", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cap.release()
            cv2.destroyAllWindows()
            success = recognizer.enroll(name, frame)
            if success:
                print(f"\n'{name}' enrolled from webcam!")
            else:
                print(f"\nNo face detected — try again with better lighting.")
            return

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\nEnrollment cancelled.")
            return

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Enroll authorized persons")
    parser.add_argument("--name",   required=True, help="Person's full name")
    parser.add_argument("--image",  help="Path to face image")
    parser.add_argument("--webcam", action="store_true", help="Capture from webcam")
    parser.add_argument("--list",   action="store_true", help="List all enrolled persons")
    parser.add_argument("--remove", help="Remove a person by name")
    args = parser.parse_args()

    recognizer = FaceRecognizer()
    recognizer.load()

    if args.list:
        persons = recognizer.list_authorized()
        print(f"\nAuthorized persons ({len(persons)}):")
        for p in persons:
            print(f"   • {p}")
        return

    if args.remove:
        recognizer.remove(args.remove)
        return

    if args.webcam:
        enroll_from_webcam(recognizer, args.name)
    elif args.image:
        enroll_from_image(recognizer, args.name, args.image)
    else:
        print("Please provide --image or --webcam")


if __name__ == "__main__":
    main()