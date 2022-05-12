#Used to extract frames from a video to images
import os
import cv2
import numpy as np

def split():
    frame_num = 0
    vid = cv2.VideoCapture("pred.mp4")
    while True:
        ret, frame = vid.read()
        frame_num += 1
        if ret == True:
            #cv2.imshow('Frame',frame)
            if frame_num % 2 == 0: #every second frame saved as an image
                filename = f'pred{frame_num}.png'
                cv2.imwrite(filename,frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print('Video has ended or failed, try a different video format!')
            break

def flipImg():
    for img  in os.listdir("C:/Users/peda_/Desktop/test/fallClose"):
        src = cv2.imread("C:/Users/peda_/Desktop/test/fallClose/"+img)
        image = cv2.flip(src, 1)
        cv2.imwrite(("C:/Users/peda_/Desktop/test/fallClose/"+img),image)


if __name__ == '__main__':
    #split()
    flipImg()