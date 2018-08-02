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
import imagetool
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
            if len(data) == 0:
                continue
            for [x, y, w, h] in data:
                mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
                #cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)

            cv2.imwrite(cleanName,img)
            cv2.imwrite(maskName, mask)

if __name__ == '__main__':
    main(sys.argv[1])