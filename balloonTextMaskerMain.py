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


def main(string):
    if not os.path.isdir('./cleaned'):
        os.mkdir('./cleaned')
    if not os.path.isdir('./mask'):
        os.mkdir('./mask')
    textFinder = ballTextMasker.BalloonCleaner()
    retrain_run_opencv.create_graph()
    for root, dirs, files in os.walk(string):
        for fname in files:
            fileName = os.path.join(root, fname)
            cleanName = 'cleaned\\' + fileName.split('\\')[-1]
            maskName = 'mask\\' + fileName.split('\\')[-1]
            img = cv2.imread(fileName)
            mask = np.zeros(img.shape,np.uint8)
            data = bubbleFinder.bubbleFinder(img)
            for i in range(len(data)):
                for j in range(i+1,len(data)):
                    if data[i][3]*data[i][4] > data[j][3]*data[j][4]:
                        data[i], data[j] = data[j], data[i]

            for [i, x, y, w, h] in data:
                forCNN = imagetool.imgResizer(img[y:y + h, x:x + w])
                if retrain_run_opencv.run_inference_on_image(forCNN) == 'goodcrop':
                    #cv2.imshow("good",forCNN)
                    #cv2.waitKey(0)
                    print('%d is deleted by CNN'%i)
                    continue
                #else:
                    #cv2.imshow("bad",forCNN)
                    #cv2.waitKey(0)
                mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
                #cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)

            cv2.imwrite(cleanName,img)
            cv2.imwrite(maskName, mask)

if __name__ == '__main__':
    main(sys.argv[1])