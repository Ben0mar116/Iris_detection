import mediapipe as mp
import cv2
import copy
import itertools
class eyeTracker():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.mpFACE = mp.solutions.face_mesh
        self.drawing_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)



    def eyeFinder(self ,image , draw = False):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #prepare image
        face_mesh = self.mpFACE.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.5 ,   max_num_faces=100)
        self.results = face_mesh.process(imageRGB) # get results -> landmarks of all faces detected
        image.flags.writeable = True
        if self.results.multi_face_landmarks and  draw: 
            
                for face_landmarks in self.results.multi_face_landmarks:
                        self.mpDraw.draw_landmarks(
                                    image=image,
                                    landmark_list=face_landmarks,
                                    # connections=FACE_CONNECTIONS,
                                    landmark_drawing_spec=self.drawing_spec,
                                    connection_drawing_spec=self.drawing_spec)
                        # draw hand landmarks
                        
        return image, self.results
    
    
    def positionFinder(self , image):
      
        if self.results.multi_face_landmarks:
            for face_no , face in enumerate(self.results.multi_face_landmarks):
                print(f"##########################face nbr : {face_no}############################")
                # for id , lm in enumerate(face.landmark):
                h, w ,c= image.shape    # height width and c is channel 
                #c will always = 3 because it represents RGB values of a pixel // the shape of an image is 3 dimentionnal tensor (height , width , c== 3 ) 
                print (f"---------------id NO : 0----------------")
                print("relative_x" , int(face.landmark[0].x * w))
                print("relative_y" , int(face.landmark[0].y * h))
                print("relative_z" , face.landmark[0].z)

    def iris_catcher (self ,image) :
        if self.results.multi_face_landmarks : 
            for face_no , face in enumerate(self.results.multi_face_landmarks):
                    
                    # RIGHT EYE

                    
                    R_eye_top    = int(face.landmark[257].y * image.shape[0])
                    R_eye_left   = int(face.landmark[362].x * image.shape[1])
                    R_eye_bottom = int(face.landmark[374].y * image.shape[0])
                    R_eye_right  = int(face.landmark[263].x * image.shape[1])

                    rightEyeHeight, rightEyeWidth, _ = image[R_eye_top:R_eye_bottom, R_eye_left:R_eye_right].shape



                    # #  Draw right eye

    
                    cloned_image = image.copy()
                    cropped_right_eye = cloned_image[R_eye_top:R_eye_bottom, R_eye_left:R_eye_right]
                    R_h, R_w, _ = cropped_right_eye.shape
                    R_x = R_eye_left
                    R_y = R_eye_top


                    xRightEye, yRightEye, rightEyeWidth, rightEyeHeight = R_x, R_y, R_w, R_h
                    cv2.rectangle(image, (xRightEye, yRightEye),
                                  (xRightEye + rightEyeWidth, yRightEye + rightEyeHeight), (200, 21, 36), 2)
    


                    # LEFT EYE
                    L_eye_top    = int(face.landmark[159].y * image.shape[0])
                    L_eye_left   = int(face.landmark[33].x * image.shape[1])
                    L_eye_bottom = int(face.landmark[145].y * image.shape[0])
                    L_eye_right  = int(face.landmark[133].x * image.shape[1])

                    leftEyeHeight, leftEyeWidth, _ = image[L_eye_top:L_eye_bottom, L_eye_left:L_eye_right].shape
                    # # draw LEFT EYE
                    cloned_image = image.copy()
                    cropped_left_eye = cloned_image[L_eye_top:L_eye_bottom, L_eye_left:L_eye_right]
                    L_h, L_w, _ = cropped_left_eye.shape
                    L_x = L_eye_left
                    L_y = L_eye_top
                    

                    xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight = L_x , L_y , L_w , L_h 
                    cv2.rectangle(image, (xLeftEye, yLeftEye),
                                  (xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (200, 21, 36), 2)
 
                
    # Crop the right eye region

    # Get the right eye coordinates on the actual -> to visualize the bbox
    



  