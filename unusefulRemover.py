#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      JeongHyeon
#
# Created:     11-05-2018
# Copyright:   (c) JeongHyeon 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import os
import cv2
import numpy as np
import scanned

def main(dir = './remained'):
     for root, dirs, files in os.walk(dir):
        for fname in files:
            if fname[:8] == 'cleaned_':
                fileName = os.path.join(root, fname)
                maskPath = fileName.replace('cleaned_','mask_')
                img = cv2.imread(fileName,cv2.IMREAD_COLOR)
                print(fname)
                if img.shape[1] > 1300:
                    print('wide image')
                    os.remove(fileName)
                    os.remove(maskPath)
                elif abs(int(img[100,100,0]) - int(img[100,100,2])) > 50:
                    print('color image')
                    os.remove(fileName)
                    os.remove(maskPath)
                elif scanned.isScanned(img) is True:
                    print('scanned image')
                    os.remove(fileName)
                    os.remove(maskPath)


if __name__ == "__main__" :
    if len(sys.argv) == 1:
        dir = './remained'
    else:
        dir = sys.argv[1]
    main(dir)
