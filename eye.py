import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


from eye_module import eyeTracker




def main() : 
    model_path = '/absolute/path/to/face_landmarker.task'

    cTime = 0 
    pastTime = 0
    cap = cv2.VideoCapture(0)
    tracker = eyeTracker()
    


    while cap.isOpened():
        cTime = time.time()
        fps = 1/(cTime - pastTime)
        pastTime = cTime
        k = cv2.waitKey(1)
        success, image = cap.read()
        # get landmarks


        # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image  , results = tracker.eyeFinder(image)
        # tracker.positionFinder(image)
        tracker.iris_catcher(image)

        # Visualize the Left and Region by drawing a rectangle on it on the actual image.
        
        # RIGH EYE
        
        cv2.putText( image ,str(int(fps)) ,(20,30) , cv2.FONT_HERSHEY_PLAIN ,  2 ,(0 ,0 ,0) ,2 )
        cv2.imshow("Camera_Capture", image)
        # FPS
        if k%256 == 27:
            # ESC pressed
            print("END...")
            break
    cap.release()

    cv2.destroyAllWindows()

if __name__ == main():
    main()
