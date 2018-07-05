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
import cv2
import numpy as np
import ballTextMasker

windowH = 720
cover = windowH//3
mainShowArea = windowH - cover

ix, iy = -1,-1
drawing = False
img = []
origin = []
back = []
mask = []
maskBack = []

def textDelete(event, x,y, flags, param):
    global ix, iy, drawing, img, origin, back, mask, maskBack

    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = origin.copy()
            cv2.rectangle(img,(ix,iy),(x,y),(255,0,0),1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img = origin.copy()
        textFinder= ballTextMasker.BalloonCleaner()
        if ix < x:
            x1 = ix
            x2 = x
        else:
            x1 = x
            x2 = ix
        if iy < y:
            y1 = iy
            y2 = y
        else:
            y1 = y
            y2 = iy
        maskBack = mask.copy()
        maskTemp, img[y1:y2+1,x1:x2+1] = textFinder.cleanBalloon(img[y1:y2+1,x1:x2+1])
        mask[y1:y2 + 1, x1:x2 + 1] = cv2.add(mask[y1:y2+1,x1:x2+1],maskTemp)
        back = origin.copy()
        origin = img.copy()
        #cv2.rectangle(img,(ix,iy),(x,y),(255,0,0),-1)
        #print(x-ix,y-iy)

def main(srcpath,dstpath) :
    global img, origin, back, mask, maskBack
    Image=cv2.imread(srcpath,cv2.IMREAD_COLOR)
    Mask = np.zeros(Image.shape,np.uint8)
    roiNum = 1
    while True:
        if Image.shape[0] <= mainShowArea*(roiNum-1):
            break
        elif Image.shape[0] > mainShowArea*roiNum + cover:
            img = Image[mainShowArea*(roiNum-1):mainShowArea*roiNum + 1 + cover]
            mask = Mask[mainShowArea*(roiNum-1):mainShowArea*roiNum + 1 + cover]
        else:
            img = Image[mainShowArea*(roiNum-1):]
            mask = Mask[mainShowArea*(roiNum-1):]

        origin = img.copy()
        back = origin.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',textDelete)

        while True:
            cv2.imshow('image',img)
            #cv2.imshow('mask',mask)
            k = cv2.waitKey(1)

            if k == 27:
                cv2.destroyAllWindows()
                if Image.shape[0] > mainShowArea*roiNum + cover:
                    Image[mainShowArea*(roiNum-1):mainShowArea*roiNum + 1 + cover] = origin
                    Mask[mainShowArea*(roiNum-1):mainShowArea*roiNum + 1 + cover] = mask
                else:
                    Image[mainShowArea*(roiNum-1):] = origin
                    Mask[mainShowArea*(roiNum-1):] = mask
                break
            elif k == 26:
                origin = back.copy()
                img = origin.copy()
                mask = maskBack.copy()

        roiNum += 1

    cleanName = dstpath + "_clean.png"
    maskName = dstpath + "_mask.png"

    cv2.imwrite(cleanName,Image)
    cv2.imwrite(maskName,Mask)


if __name__ == "__main__" :
    main(sys.argv[1])
