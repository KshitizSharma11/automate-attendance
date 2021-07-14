import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'C:\\Users\\hp\\Downloads\\records'
images =[]
classnames =[]
mylist =os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
def markattendance(name):
    with open('C:\\Users\\hp\\Documents\\attendance.csv','r+') as f:
        myDatalist = f.readlines()
        namelist =[]


        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S') 
            f.writelines(f'\n{name},{dtstring}')   


encodelistknown = findEncodings(images)
print('encoding complete')


cap = cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    imgS =cv2.resize(img,(0,0),None,0.25,0.25)
    imgS =cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)


    facesCur = face_recognition.face_locations(imgS)
    encodeCur = face_recognition.face_encodings(imgS,facesCur)

    for encodeface,faceloc in zip(encodeCur,facesCur):
        matches =face_recognition.compare_faces(encodelistknown,encodeface)
        dist= face_recognition.face_distance(encodelistknown,encodeface)
        matchindex = np.argmin(dist)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            y1,x2,y2,x1 =faceloc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            markattendance(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)        


    


