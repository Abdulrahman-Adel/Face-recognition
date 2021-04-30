# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:15:12 2021

@author: Abdelrahman
"""

import cv2
import numpy as np
import os

def  V2Fs(Video_path,person_name, counter, required_size = (160, 160)):
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
    A dataset of extracted faces from each frame with it's respective label (persons' name).
    '''
    
    cap = cv2.VideoCapture(Video_path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    i = counter
    
    X_train = []
    y_train = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = np.asarray(image)
        
        faces = face_cascade.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)
        #To save only frames with one detectable face in them
        if(len(faces) == 1):
            x, y, h, w = faces[0]
            roi = pixels[y:y+h, x:x+w]    
            image = cv2.resize(roi, required_size)
            cv2.imwrite("images/"+person_name+"_"+str(i)+".jpg", image)
            face_array = np.asarray(image)
            X_train.append(face_array)
            y_train.append(person_name)
            i += 1
    

    cap.release()
    cv2.destroyAllWindows()
    
    return X_train, y_train, i


    
Base_dir = os.path.dirname(os.path.abspath(__file__))   
Video_dir = os.path.join(Base_dir, "Videos")
X, y = [], []

for folder in os.listdir(Video_dir):
    folder_path = os.path.join(Video_dir, folder)
    counter = 0
    for file in os.listdir(folder_path):
        if file.endswith("mp4"):
            file_path = os.path.join(folder_path, file)
            trainX, trainy, counter = V2Fs(file_path, folder, counter)
           
            X.extend(trainX)
            y.extend(trainy)
           

#Saving the dataset
np.savez_compressed('Face_authen.npz', X, y)
print("Dataset Saved !!")
        
           








