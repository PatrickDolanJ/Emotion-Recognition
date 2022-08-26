# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
from deepface import DeepFace
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


video = cv2.VideoCapture(0)
while video.isOpened():
    _,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for x,y,w,h in face:
        img=cv2.rectangle(frame,(x,y),(x+w,y+w),(0,0,255),1)
        try:
            analyze=DeepFace.analyze(frame,actions=['emotion'])
            print(analyze['dominant_emotion'])
        except:
            print("no face detected")

        cv2.imshow('video',frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
            video.release()



img_path = 'face2.jpg'
image = cv2.imread(img_path)

analyze = DeepFace.analyze(image, actions=['emotion'])
print(analyze)









def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
