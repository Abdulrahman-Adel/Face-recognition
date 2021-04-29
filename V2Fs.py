# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:15:12 2021

@author: Abdelrahman
"""

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os

def  V2Fs(Video_path,person_name, required_size = (160, 160)):
    '''
    Parameters
    ---------- 
    Video_path : String
    person_name : String
    required_size : tuple
    
    DESCRIPTION
    -----------
    A function that extracts images from a given video frames
    , save it in a folder with the same name as the video name
    and create a dataset.
    
    Returns
    -------
    None.
    '''
    
    cap = cv2.VideoCapture(Video_path)
    detector = MTCNN()
    
    i = 0
    
    X_train = []
    y_train = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = np.asarray(image)
        
        faces = detector.detect_faces(pixels)
        #To save only frames with one detectable face in them
        if(len(faces) == 1):
            x, y, h, w = faces[0]['box']
            roi = pixels[y:y+h, x:x+w]    
            image = cv2.resize(roi, required_size)
            cv2.imwrite("images/"+person_name+"_"+str(i)+".jpg", image)
            face_array = np.asarray(image)
            X_train.append(face_array)
            y_train.append(person_name)
            i += 1
    

    cap.release()
    cv2.destroyAllWindows()
    
    return X_train, y_train


    
Base_dir = os.path.dirname(os.path.abspath(__file__))   

X, y = [], []

for root, dirs, files in os.walk(Base_dir):
    for file in files:
        if file.endswith("mp4"):
           file_path = os.path.join(root, file)
           trainX, trainy = V2Fs(file_path, file[:-4])
           
           X.extend(trainX)
           y.extend(trainy)
           

#Saving the dataset
np.savez_compressed('Face_authen.npz', X, y) 

        
           








