# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:36:45 2021

@author: Abdelrahman
"""

import cv2
import numpy as np
import face_recognition

"""face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Abdulrahman.yml")"""

img = cv2.imread("WIN_20210428_22_43_12_Pro.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(img)[0]

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame_by_frame
    _, frame = cap.read()
    
    #Converting to Grayscale
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Detecting faces
    faces = face_recognition.face_locations(frame_rgb)
    Encoding = face_recognition.face_encodings(frame_rgb, faces)
    
    #Iterating through faces
    for faceLoc, faceEnc in zip(faces, Encoding):
        
        y1, x2, y2, x1 = faceLoc
        #Predicting
        result = face_recognition.compare_faces([img_encoding], faceEnc)
        faceDis = face_recognition.face_distance([img_encoding], faceEnc)
        
        if result[list(faceDis).index(min(faceDis))] and min(faceDis) < 0.4:
            #Drawing A rectangle
            color = (0, 255, 0)
            stroke = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, stroke)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, "Abdulrahman", (x1+6, y2-6),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            color = (0, 0, 255)
            stroke = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, stroke)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, "Stranger", (x1+6, y2-6),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
  
cap.release()
cv2.destroyAllWindows()    