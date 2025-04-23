import os
import cv2 as cv
import numpy as np
#import opencv-contrib-python as ocp




haar_cascade = cv.CascadeClassifier("C:\\Users\\produ\\Desktop\\CIJ\\Code_py\\haar_face.xml")

features = []
#labels = []
path=r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\frames'

def create_train():
    for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                #labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
#labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
#face_recognizer.train(features,labels)
face_recognizer.train(features)
if os.path.exists(r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\face_trained.yml'):
    os.remove(r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\face_trained.yml')

face_recognizer.save(r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\face_trained.yml')
np.save(r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\features.npy', features)
#np.save(r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\labels.npy', labels)
