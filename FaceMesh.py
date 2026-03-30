import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import time
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


class FaceMesh:
    def __init__(self):
        self.model_path="face_landmarker.task"

        self.options=vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            num_faces=1
        )
        self.mpFace=vision.FaceLandmarker.create_from_options(self.options)
        self.mpDraw=drawing_utils
        self.drawing_spec=self.mpDraw.DrawingSpec(circle_radius=1,thickness=1)
        self.connections=vision.FaceLandmarksConnections
        
        self.FACE_COLOR = (192, 192, 192)   
        self.LEFT_EYE_COLOR = (51, 25, 0)   
        self.RIGHT_EYE_COLOR = (25, 51, 0)  
        self.LIPS_COLOR = (153, 153, 255)   
        self.IRIS_COLOR = (153, 255, 255)   

    def detect(self,img,Draw=True):
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mpImg=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        results=self.mpFace.detect(mpImg)
        lmList=results.face_landmarks
        det=lmList[0]
        if Draw:
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.FACE_COLOR, thickness=1)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255,255,255), thickness=1)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.LEFT_EYE_COLOR, thickness=2)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.RIGHT_EYE_COLOR, thickness=2)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.LIPS_COLOR, thickness=2)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.IRIS_COLOR, thickness=2)
            )
            self.mpDraw.draw_landmarks(
                img,
                landmark_list=det,
                connections=self.connections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.IRIS_COLOR, thickness=2)
            )
        face=[]
        for id,lm in enumerate(det):
            h,w,c=img.shape
            x,y=int(lm.x*w),int(lm.y*h)
            face.append((id,x,y))
        return img,face

def main():
    cap= cv2.VideoCapture(0)
    ptime=0
    detector=FaceMesh()
    while True:
        ret,img=cap.read()
        img=cv2.flip(img,1)
        
        img,face=detector.detect(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,f"FPS: {int(fps)}",(50,70),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Face Mesh",img)
        if cv2.waitKey(1)& 0xFF==27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()