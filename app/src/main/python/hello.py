import cv2
import numpy as np
import copy
import imutils
folc= np.zeros((800,500))
# print(type(folc))
ext=(0,0)
extr=(0,0)
b=[]
img = np.zeros((800,500,3),dtype=np.uint8)
def test(red,blue,green):
    mask = np.zeros((800, 500, 3), dtype=np.uint8)
    for i in range(0, 500):
        for j in range(0, 800):
            aa = blue[i][j]
            bb = green[i][j]
            cc = red[i][j]
            mask[j, i] = (aa, bb, cc);


    mask = cv2.resize(mask, (500, 800))
    global img
    img=mask
    #cut paste here
    output_image=cv2.rectangle(img,(0,0),(200,200),(0,0,255),-1)
    #cut paste up

    blue, green, red = cv2.split(output_image)
    return red, green, blue

