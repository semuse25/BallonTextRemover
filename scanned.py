#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      JeongHyeon
#
# Created:     13-07-2018
# Copyright:   (c) JeongHyeon 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import cv2
import numpy as np


def isScanned(img):
    if img is None:
        return True
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.plot(hist/(img.shape[0]*img.shape[1]))
    #plt.show()
    val = np.where(hist == max(hist))[0]
    rate = np.sum(hist[15:240])/(img.shape[0]*img.shape[1])
    if rate > 0.95:
        print('rate is ',rate)
        return True
    else:
        return False
    #print(val)
    #if (val < 240).any() and (val > 15).any():
    #    print('rate is ',rate)
    #    return True
    #else:
    #    return False

