#!/usr/bin/etc python

import cv2
import numpy as np
from matplotlib import pyplot as plt


class TextFinder:

    def deleteCanny(self,canny, cnt):
        img_h, img_w = canny.shape

        for i in cnt:
            canny[i[0,1],i[0,0]] = 0

    def findColor(self,img,color):
        arr = np.where(img == color)[0]
        for i in range(len(arr)-3):
            if(arr[i] == arr[i+2]):
                return True
        return False

    def FindText(self,img):
        blackMargin =5
        original = img.copy()
        img_h,img_w = img.shape[:2]
        frame = np.zeros((img_h+(2*blackMargin),img_w+(2*blackMargin),3), np.uint8)
        frame[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin] = img

        bin = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(bin,176,255,cv2.THRESH_BINARY)
        for i in bin:
            if np.count_nonzero(i) > 2:
                i = [255,255,255]
            else:
                i = [0,0,0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,4))
        bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        bin = cv2.erode(bin,kernel,iterations = 1)


        blur = cv2.GaussianBlur(bin,(3,3),0)
        canny=cv2.Canny(blur,200,300)

        cnts,contours,hierarchy  = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return original

        for i in range(len(contours)):
            cnt=contours[i]
            if cv2.arcLength(cnt, True) < img_h + img_w:
                self.deleteCanny(canny,cnt)


        while True:
            cnts,contours,hierarchy  = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if len(contours) == 0:
                return original

            area, contour = 0, [0]

            for i in range(len(contours)):
                cnt=contours[i]
                if area < cv2.contourArea(cnt):
                    area = cv2.contourArea(cnt)
                    contour[0] = cnt

            if area < img_h*img_w//5:
                canny = cv2.dilate(canny,kernel,iterations = 1)
            else:
                break


        point = contour[0][0]
        bgColor = frame[point[0,1]+1,point[0,0]+1]
        bgColor = bgColor.tolist()
        cv2.drawContours(frame, contour, -1, bgColor, -1)

        return frame[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin]

##textFinder=TextFinder()
##
##for i in range(1,51):
##    Image = 'ball' + str(i) +'.jpg'
##    cnImage = 'canny' + str(i) +'.jpg'
##    outputImage = 'found' + str(i) +'.jpg'
##    img=cv2.imread(Image,cv2.IMREAD_COLOR)
##    cny, findImg = textFinder.FindText(img)
##    cv2.imwrite(cnImage,cny)
##    cv2.imwrite(outputImage,findImg)





