import cv2
import numpy as np
from matplotlib import pyplot as plt

def deleteCanny(canny, cnt):
    img_h, img_w = canny.shape

    for i in cnt:
        canny[i[0,1],i[0,0]] = 0

class BalloonCleaner:

    def cleanBalloon(self, image):
        if image is None:
            return -1
        img = image.copy()
        blackMargin = 5
        img_h,img_w = img.shape[:2]
        frame = np.zeros((img_h+(2*blackMargin),img_w+(2*blackMargin),3), np.uint8)
        mask = np.zeros((img_h+(2*blackMargin),img_w+(2*blackMargin),3), np.uint8)
        frame[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin] = img

        bin = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(bin,220,255,cv2.THRESH_BINARY)

        # remove noise
        img = cv2.GaussianBlur(bin,(3,3),0)

        # convolute with proper kernels
        laplacian = cv2.Laplacian(img,cv2.CV_8U)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        laplacian = cv2.dilate(laplacian,kernel,iterations = 1)

        cnts,contours,hierarchy  = cv2.findContours(laplacian, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print("none!")

        for i in range(len(contours)):
            cnt=contours[i]
            if cv2.arcLength(cnt, True) < (img_h + img_w):
                deleteCanny(laplacian,cnt)



        cnts,contours,hierarchy  = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = img_h*img_w
        n = 0
        hie = hierarchy[0]
        for i in range(len(hie)):
            if cv2.contourArea(contours[i]) > (img_h * img_w)*0.25:
                if cv2.contourArea(contours[i]) < area:
                   area = cv2.contourArea(contours[i])
                   n = i

        mask2 = mask.copy()
        cv2.drawContours(mask, contours[n], -1, (255,255,255), 2)
        grayMask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        cn,co,hi  = cv2.findContours(grayMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areaTemp = img_h*img_w
        nTemp = 0
        for i in range(len(hi[0])):
            if cv2.contourArea(co[i]) > (img_h * img_w)*0.25:
                if cv2.contourArea(co[i]) < areaTemp:
                   area = cv2.contourArea(co[i])
                   nTemp = i
        cv2.drawContours(mask2, co[nTemp], -1, (255,255,255), 6)

        mask = mask2.copy()

        floodflags = 8
        floodflags |= (255 << 8)
        floodMask = np.zeros((img_h+(2*blackMargin)+2,img_w+(2*blackMargin)+2), np.uint8)
        cv2.floodFill(mask, floodMask, (0,0), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (0,img_h+(2*blackMargin)-1), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (img_w+(2*blackMargin)-1,0), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (img_w+(2*blackMargin)-1,img_h+(2*blackMargin)-1), (255,255,255),(0,0,0),(0,0,0),floodflags)

        mask = cv2.bitwise_not(mask)

        hImg = cv2.bitwise_and(frame,mask)
        gray = cv2.cvtColor(hImg,cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[1,256])
        hist = hist.tolist()
        colMax = max(hist)
        bgColor = hist.index(colMax)


        colMask = np.full((img_h+(2*blackMargin),img_w+(2*blackMargin),3),bgColor, np.uint8)
        maskColMask = np.full((img_h+(2*blackMargin),img_w+(2*blackMargin),3),(0,0,255), np.uint8)
        maskInv = cv2.bitwise_not(mask)
        mask = cv2.bitwise_and(colMask,mask)

        textMask = cv2.bitwise_and(maskColMask,mask)
        textMask = cv2.bitwise_not(textMask)
        textMask = cv2.add(textMask,frame)
        textMask = cv2.bitwise_not(textMask)

        frame = cv2.bitwise_and(frame,maskInv)
        frame = cv2.add(frame,mask)
        return textMask[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin], frame[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin]


