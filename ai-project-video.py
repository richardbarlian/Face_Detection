import cv2
import os
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Check if we run Windows (NT)
if os.name == 'nt':
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#In case we run on Linux/MacOS
else:
    webcam = cv2.VideoCapture(0)
    

#Loops to find face in every frame
while True:
    succesful_frame_read, frame = webcam.read()

    #Changes it to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Lets AI detect it
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draws rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)

    #Shows picture
    cv2.imshow('Clever Programmer Face Detector', frame)

    #Waits 1 millisecond before next frame
    key = cv2.waitKey(1)
    
    #Quits if user types 'q' or 'Q'
    if key==81 or key==113:
        break

webcam.release()
    
print("Code Completed")