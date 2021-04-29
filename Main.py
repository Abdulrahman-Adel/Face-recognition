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
    
    detector = MTCNN()
    
    results = detector.detect_faces(pixels)
    
    faces_arrays = []
    for bb in results:
        x1, y1, h, w = bb['box']
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        
        face = pixels[y1:y2, x1:x2]
        
        image = cv2.resize(face, required_size)
        face_array = np.asarray(image)
        
        faces_arrays.append(face_array)
        
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
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    
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
    face_embeddings = get_embedding(face)
    
    face_embeddings = np.expand_dims(face_embeddings, axis = 0)
    
    clf = pickle.load(open("finalized_model.sav", 'rb'))
    
    yhat = clf.predict(face_embeddings)
    yprob = clf.predict_proba(face_embeddings)
    print("yhat :", yhat) #--> yhat : [0]
    print("yprob :", yprob) #--> yprob : [[9.9999990e-01 1.0000001e-07]]
    
if __name__ == "__main__":
    model = load_model("facenet_keras.h5")
    
    Test()



    
    
    
    
