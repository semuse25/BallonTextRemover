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
    #print(val)
    if val < 230 and val > 30:
        return True
    else:
        return False

