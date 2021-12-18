import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# Function os.scandir() iterates through a folder
# Providing the path of the directory
# r = raw string literal
dirloc = r"C:\Users\dinok\Documents\Dissertetion\Dataset\Seperated"




#Face cascade and eye_cascade objects
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')




# calling scandir() function
for file in os.scandir(dirloc):
    if(file.path.endswith(".mp4")) and file.is_file():
        # Create a VideoCapture object and read from input file
        # cap = cv2.VideoCapture(file.path)------
        cap = cv2.VideoCapture(file.path)
        
        
        
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Default resolutions of the frame are obtained. The
        # default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        print('Width %d'%frame_width)
        print('Height %d'%frame_height)
        print('fps %d'%fps)

        # Define the codec and create VideoWriter object.The
        # output is stored in 'outpy.avi' file.
        #out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),10, (frame_width, frame_height))

        # Read until video is completed
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                #Run facial detection
                #Converting image to grayscale

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                #Creatubg variable faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                #Defining and drawing the rectangle around the face

                for(x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3)
                
                #Creating two regions of interest
                roi_gray= gray[y:(y+h), x:(x+w)]
                roi_color= frame[y:(y+h),x:(x+w)]

                #Creating variable eyes
                eyes= eye_cascade.detectMultiScale(roi_gray, 1.1,4)
                index= 0

                #Creating for loop in order to divide one eye from another
                for(ex, ey, ew, eh) in eyes:
                    if index==0:
                        eye_1 = (ex, ey, ew, eh)
                    elif index ==1:
                        eye_2 = (ex, ey, ew, eh)

                    #Drawing rectangles around the eyes
                    cv2.rectangle(roi_color,(ex,ey) , (ex+ew, ey+eh), (0,0,255), 3)
                    index = index+1
                
                if eye_1[0] < eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                #Calculating coordinates of a central points of the rectangles
                left_eye_center = (int(left_eye[0] + (left_eye[2]/2)), int(left_eye[1] + (left_eye[3]/2)))
                left_eye_x = left_eye_center[0]
                left_eye_y = left_eye_center[1]

                right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
                right_eye_x = right_eye_center[0]
                right_eye_y = right_eye_center[1]

                cv2.circle(roi_color, left_eye_center, 5, (255,0,0) , -1)
                cv2.circle(roi_color, right_eye_center, 5, (255,0,0), -1)
                cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200), 3)

                if left_eye_y > right_eye_y:
                    A = (right_eye_x, left_eye_y)
                    #Integer -1 indicates that the image will rotate in the clockwise direction
                    direction = -1
                
                else:
                    A = (left_eye_x, right_eye_y)
                    #Integer 1 indicates that image will rotate in the counter clockwise
                    #direction
                    direction = 1

                cv2.circle(roi_color, A, 5, (255,0,0), -1)

                cv2.line(roi_color, right_eye_center, left_eye_center, (0,200,200), 3)
                cv2.line(roi_color,left_eye_center, A, (0,200,200),3)
                cv2.line(roi_color, right_eye_center, A, (0,200,200), 3)

                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y
                angle = np.arctan(delta_y/delta_x)
                angle = (angle * 180) / np.pi

                #Width and height of the video
                h, w = frame.shape[:2]
                
                #Calculating a center point of the video
                #Integer division "//" ensures that we receive whole numbers
                center = (w // 2, h //2)
                
                #Defining a matrix M and calling
                #cv2.getRotationMatrix2D method
                M = cv2.getRotationMatrix2D(center, (angle), 1.0)
                
                #Applying the rotation to our video using the cv2.warpAffine method
                rotated = cv2.warpAffine(frame, M , (w, h))
            


                #cv2.imshow('mat', frame)
                cv2.imshow('mat',rotated)
                
                #Rotate Crop the image and scale to 256x256



                # Write the frame into the file 'output.avi'
                #out.write(frame)

                # Display the resulting frame
                #cv2.imshow('frame', frame)
                #cv2.waitKey(0) number inside waitKey is equal to the time in milliseconds
                #we want each frame to be displayed

                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
        #out.release()
        # Closes all the frames
        cv2.destroyAllWindows()
