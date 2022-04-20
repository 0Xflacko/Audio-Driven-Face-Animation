import face_alignment
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from skimage import io
import collections


# Function os.scandir() iterates through a folder
# Providing the path of the directory
# r = raw string literal
dirloc = r"C:\Users\dinok\Documents\Dissertetion\Dataset\Seperated"
s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset\clip\images"

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')


def fun(det):
    return [tuple(i) for i in det]

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized



i = 0
j = 0

# calling scandir() function
for file in os.scandir(dirloc):
    if(file.path.endswith(".mp4")) and file.is_file():
        # Create a VideoCapture object and read from input file
        # cap = cv2.VideoCapture(file.path)------
        cap = cv2.VideoCapture(file.path)
        
        print(file)
        
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Default resolutions of the frame are obtained. The
        # default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        #print('Width %d'%frame_width)
        #print('Height %d'%frame_height)
        #print('fps %d'%fps)

        # Define the codec and create VideoWriter object.The
        # output is stored in 'outpy.avi' file.
        #out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),10, (frame_width, frame_height))
        i = 1
        j = 0
        # Read until video is completed
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#Changes the colour space to RGB
                
                #Run facial detection

                det = fa.get_landmarks_from_image(frame_temp)#Process entire directory in one go

               ## plot_style = dict(marker='o',
                  ##                  markersize=4,
                    ##                linestyle='-',
                      ##              lw=2)

              ##  det_type = collections.namedtuple('prediction_type', ['slice', 'color'])
              ##  det_types = {'face': det_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                ##            'eyebrow1': det_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                  ##          'eyebrow2': det_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                    ##        'nose': det_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                      ##      'nostril': det_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                        ##    'eye1': det_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                          ##  'eye2': det_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                            ##'lips': det_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                            ##'teeth': det_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                           ## }

                
                ##fig = plt.figure(figsize=plt.figaspect(.5))
                ##ax = fig.add_subplot(1,2,1)
                ##ax.imshow(frame)

                ##for det_type in det_types.values():
                  ##  ax.plot(det[det_type.slice,0],
                    ##        det[det_type.slice,1],
                      ##      color=det_type.color, **plot_style)

                ##ax.axis('off')
                index = 0
                #for detection in det[0]:
                   # print(detection[0])
                   # cv2.circle(frame,(int(detection[0]), int(detection[1])), 10, (255,0,0), -1)
                    #cv2.putText(frame, '%0d'%index, (int(detection[0]), int(detection[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1 ,(0,0,255), 2)
                   # index = index+1

                

                #Get the average of a list
                def Average(lst):
                  return sum(lst) / len(lst)

                lst_left_eye = [det[0][36], det[0][37], det[0][38], det[0][39], det[0][40], det[0][41]]
                lst_right_eye = [det[0][42],det[0][43], det[0][44], det[0][45], det[0][46], det[0][47]]

                

                average_left_eye = Average(lst_left_eye)
                average_right_eye = Average(lst_right_eye)
                #print("Average of the list = ", average_left_eye)
                #print("Average of the list = ", average_right_eye)

                left_eye_x = average_left_eye[0]
                left_eye_y = average_left_eye[1]

                right_eye_x = average_right_eye[0]
                right_eye_y = average_right_eye[1]

                if left_eye_y > right_eye_y:
                    A = (right_eye_x, left_eye_y)
                    #Integer -1 indicates that the image will rotate in the clockwise direction
                    direction = -1

                else:
                    A = (left_eye_x, right_eye_y)
                    #Integer 1 indicates that image will rotate in the counter clockwise
                    #direction
                    direction = 1   

                

                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y

                angle = np.arctan(delta_y/delta_x)
                angle = (angle * 180) / np.pi

                h, w = frame.shape[:2]
                
                #Calculating a center point of the video
                #Integer division "//" ensures that we receive whole numbers
                center = (w // 2, h //2)

                #Defining a matrix M and calling
                #cv2.getRotationMatrix2D method
                M = cv2.getRotationMatrix2D(center, (angle), 1.0)
                
                #Applying the rotation to our video using the cv2.warpAffine method
                rotated = cv2.warpAffine(frame, M , (w, h))
                

                #Crop the image and scale to 256x256

                # calculate distance between the eyes in the first image
               # dist_1 = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))

                # calculate distance between the eyes in the second image
                #dist_2 = np.sqrt((delta_x_1 * delta_x_1) + (delta_y_1 * delta_y_1))

                

                dim = (256,256)
                crop_img = rotated[200:1150, 0:1080]
                #resized = crop_square(rotated, 256, interpolation=cv2.INTER_AREA)
                resized = cv2.resize(crop_img, dim,interpolation=cv2.INTER_AREA)
                #resized = cv2.resize(rotated,dim)
                cv2.imshow('mat', resized)

                
                #//SAVING PART//
                name = './Dataset/' + file.name + '/images/%04d.png'%j
                print('Creating...' + name)
                os.makedirs('./Dataset/' + file.name + '/images/', exist_ok=True)
                cv2.imwrite(name, resized)
                j += 1                   
                

                  
                    

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
