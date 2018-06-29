#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      JeongHyeon
#
# Created:     28-06-2018
# Copyright:   (c) JeongHyeon 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import os
import cv2
import numpy as np
import bubbleFinder  # github.com/sKHJ/speechBubbleFinder
import ballTextMasker


def main(string):
    if not os.path.isdir('./cleaned'):
        os.mkdir('./cleaned')
    if not os.path.isdir('./mask'):
        os.mkdir('./mask')
    textFinder = ballTextMasker.BalloonCleaner()

    for root, dirs, files in os.walk(string):
        for fname in files:
            fileName = os.path.join(root, fname)
            cleanName = 'cleaned\\' + fileName.split('\\')[-1]
            maskName = 'mask\\' + fileName.split('\\')[-1]
            img = cv2.imread(fileName)
            mask = np.zeros(img.shape,np.uint8)
            data = bubbleFinder.bubbleFinder(img)
            for [x, y, w, h] in data:
                mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)
                #shrink = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
                #cv2.imshow('process', shrink)
                #cv2.waitKey(0)
            cv2.imwrite(cleanName,img)
            cv2.imwrite(maskName, mask)

if __name__ == '__main__':
    main(sys.argv[1])
