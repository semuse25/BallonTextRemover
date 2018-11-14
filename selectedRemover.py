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
import retrain_run_opencv
import scanned

def main(string):
    retrain_run_opencv.create_graph()
    if not os.path.isdir('./cleaned'):
        os.mkdir('./cleaned')
    if not os.path.isdir('./mask'):
        os.mkdir('./mask')
    textFinder = ballTextMasker.BalloonCleaner()
    for root, dirs, files in os.walk(string):
        for fname in files:
            fileName = os.path.join(root, fname)
            cleanName = 'cleaned\\cleaned_'+fileName.split('\\')[-1]
            #resName = 'cleaned\\res_' + fileName.split('\\')[-1]
            maskName = 'cleaned\\mask_' +fileName.split('\\')[-1]
            img = cv2.imread(fileName)
            mask = np.zeros(img.shape,np.uint8)
            if img is None:
                continue
##            if scanned.isScanned(img) is True:
##                print(fileName,' is scanned')
##                continue
            data = bubbleFinder.bubbleFinder(img)
            if len(data) is not 0:
                for [x, y, w, h] in data:
                    if x < 5:
                        x1 = 0
                    else:
                        x1 = x-5
                    if y < 5:
                        y1 = 0
                    else:
                        y1 = y-5
                    if x+w > img.shape[1] - 5:
                        w = img.shape[1] - x
                    else:
                        w += 5
                    if y+h > img.shape[0] - 5:
                        h = img.shape[0] - y
                    else:
                        h += 5
                    maskTemp, img[y1:y + h, x1:x + w] = textFinder.cleanBalloon(img[y1:y+h, x1:x+w])
                    mask[y1:y + h, x1:x + w] = cv2.bitwise_or(mask[y1:y + h, x1:x + w],maskTemp)
            cv2.imwrite(cleanName,img)
##            for [x, y, w, h] in data:
##                cv2.rectangle(img, (x-5, y-5), (x + w+5, y + h+5), (255, 30, 0), 3)
##            if len(badData) == 0:
##                continue
##            for [x, y, w, h] in badData:
##                cv2.rectangle(img, (x-5, y-5), (x + w+5, y + h+5), (30, 0, 255), 3)
##
##            print(len(data))
##            print(len(badData))

            #cv2.imwrite(resName,img)
            cv2.imwrite(maskName, mask)
            os.remove(fileName)

if __name__ == '__main__':
    main(sys.argv[1])