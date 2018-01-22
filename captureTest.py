from PIL import ImageGrab
import win32api as win
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

MIN = np.array([0, 0, 75],np.uint8)
MAX = np.array([255, 255, 255],np.uint8)

def evalScreen(debug):
    #Get the image and converto to numpy array
    img = ImageGrab.grab(bbox=(0,0, win.GetSystemMetrics(0), win.GetSystemMetrics(1)))
    img_np = np.array(img)


    raw = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    frameHSV = cv2.resize(raw, (300, 300)) 
    

    mask = cv2.inRange(frameHSV, MIN, MAX)
    frameHSV = cv2.bitwise_and(frameHSV, frameHSV, mask= mask)

    originalShape = frameHSV.shape

    frameHSV = frameHSV.reshape((-1, 3))
    Z= np.sort(frameHSV, axis = 0)


    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(originalShape)
    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2RGB)

    unique, counts = np.unique(res2, axis=0, return_counts=True)
    


    # if(debug):
    #     res2 = res.reshape(originalShape)
    #     res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2RGB)
    #     res2 = cv2.resize(res2, (50, 350)) 
    #     cv2.imshow('res2', res2)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    return(unique, counts)



#    cv2.destroyAllWindows()
