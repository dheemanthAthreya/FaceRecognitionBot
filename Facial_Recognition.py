import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime



path = 'TrainedImages'                                              #contains all the faces that can be identified
images = []                                                         #images[] is used to store the actual images
names = []                                                          #names[] is used to store the name of each image
myList = os.listdir(path)                                                    
for img in myList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    names.append(os.path.splitext(img)[0])


def findEncoding(images):
    """
    A function that iterates through a given list, converts all the items in it from BGR to RGB format, 
    encodes them and stores the encodings in encodeList[]

    """                                          
    encodeList = []
    for pic in images:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(pic)[0]
        encodeList.append(encode)
    return encodeList

def recordDetection(name):
    """
    A function that adds the name and time to a csv file when a face is detected.
    If the name is already present, it doesn't add it again.
    """
    with open('detectionData.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

                                                                    
encodeListKnown = findEncoding(images)                              #encodes the images stored in TrainedImages

videoFeed = cv2.VideoCapture(0)                                     #used to take input from the default webcam. Changes can be made here to take video input from different sources 
                                                                    
while True:
    """
    video feed is continuously read, resized to help with processing
    and converted from BGR to RGB format
    """
    success, img = videoFeed.read()
    #imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)                   
    imgSmall = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    faceLocCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,faceLocCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceLocCurFrame):
        """
        faces from the video feed are compared with the known faces and
        the images with the least distance between them are considered as a match
        """
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            """
            if the faces from the video feed match the known faces, a rectangle is plotted around
            the face and the name is displayed to help with identification.
            """
            name = names[matchIndex].upper()            
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1,x2,y2,x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1-40,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            recordDetection(name)                                   #the names of the matching faces are added to csv file to record the time of detection
    
    
    cv2.imshow('webcam',img)                                        #displays the video feed from the webcam
    cv2.waitKey(1)

