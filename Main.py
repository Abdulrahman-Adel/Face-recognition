# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:19:18 2021

@author: Abdelrahman
"""

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from numpy import savez_compressed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pickle

def Extract_faces(image, required_size = (160, 160)):
    '''
    Parameters
    ---------- 
    image : image array
    required_size : tuple
    
    DESCRIPTION
    -----------
    A function that extracts faces from an image.
    
    Returns
    -------
    an array of extracted faces.
    '''
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(image)
    
    #Takes so much time
    """detector = MTCNN()
    results = detector.detect_faces(pixels)""" 
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    results = face_cascade.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)
    
    faces_arrays = []
    for bb in results:
        x1, y1, h, w = bb
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        
        face = pixels[y1:y2, x1:x2]
        
        image = cv2.resize(face, required_size)
        face_array = np.asarray(image)
        
        faces_arrays.append([face_array, bb])    
    return faces_arrays  

def get_embedding(face_pixels):
    '''
    Parameters
    ---------- 
    face_pixles : image array
    
    DESCRIPTION
    -----------
    Extract embeddings from an image
    
    Returns
    -------
    embeddings of an image
    '''
    
    face_pixels = face_pixels.astype("float32")
    
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    
	# transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
    yhat = model.predict(samples)
    
    return yhat[0]

def load_dataset(dataPath):   
    '''
    Parameters
    ---------- 
    dataPath : String
    
    DESCRIPTION
    -----------
    Creating a dataset from image embeddings to train the model
    
    Returns
    -------
    an array of faces in all the image with thier respective label
    '''
    
    data = np.load(dataPath)
    
    X, y = data['arr_0'], data['arr_1']
    
    Xnew = []
    for face_pixels in X:
        embedding = get_embedding(face_pixels)
        Xnew.append(embedding) 
        
    Xnew, y = np.asarray(Xnew), np.asarray(y) 
    
    savez_compressed('Face_authen_abdul_embeddings.npz', Xnew, y) 
    
    return Xnew, y        


def Train():
    '''
    Parameters
    ---------- 
    None.
    
    DESCRIPTION
    -----------
    Training an SVC on the embedding dataset.
    
    Returns
    -------
    None
    '''
    X_train, y_train = load_dataset("Face_authen.npz")

    X_encoder = Normalizer(norm='l2')
    X_train = X_encoder.transform(X_train) 
    
    y_encoder = LabelEncoder()
    y_encoder.fit(y_train) 
    y_train = y_encoder.transform(y_train)
    
    #clf = SVC(kernel='linear', probability=True)
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    
    filename = 'finalized_model_xgb.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Model Saved !!!")
    
    
def Test():
    '''
    Parameters
    ---------- 
    None.
    
    DESCRIPTION
    -----------
    Testing the model.
    
    Returns
    -------
    None
    '''
    
    image = cv2.imread("WIN_20210428_22_43_12_Pro.jpg")
    
    face = Extract_faces(image)[0]
    face_embeddings = get_embedding(face[0])
    
    face_embeddings = np.expand_dims(face_embeddings, axis = 0)
    
    yhat = clf.predict(face_embeddings)
    yprob = clf.predict_proba(face_embeddings)
    print("yhat :", yhat) #--> yhat : [0]
    print("yprob :", yprob[0]) #--> yprob : [0.99884987 0.00115013]
    
    
def Camera_Test():
    '''
    Parameters
    ---------- 
    None.
    
    DESCRIPTION
    -----------
    Testing the model real-time on laptop camera
    
    Returns
    -------
    None.
    '''
    
    cap = cv2.VideoCapture(0)

    while(True):
        #Capture frame_by_frame
        _, frame = cap.read()
        
        #Detecting faces
        faces = Extract_faces(frame)
    
        #Iterating through faces
        for face in faces:
            
            x, y, w, h = face[1]
            
            face_Emb = get_embedding(face[0])
            face_Emb = np.expand_dims(face_Emb, axis = 0)
        
            #Predicting
            result = clf.predict(face_Emb)
            prob = clf.predict_proba(face_Emb)
           
            if result[0] == 0 and prob[0][0] > 0.99:
                #Drawing A rectangle
                color = (0, 255, 0)
                stroke = 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(frame, "Abdulrahman", (x+6, y+h-6),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                color = (0, 0, 255)
                stroke = 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(frame, "Stranger", (x+6, y+h-6),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.imshow("Face Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
  
    cap.release()
    cv2.destroyAllWindows() 
       
if __name__ == "__main__":
    model = load_model("facenet_keras.h5")
    clf = pickle.load(open("finalized_model_xgb.sav", 'rb'))
    
    #Train()
    Camera_Test()
    #Test()



    
    
    
    
