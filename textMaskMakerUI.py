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
mode = 'RECT'
img = []
origin = []
back = []
mask = []
maskBack = []
maskTemp = []
rad = 10
color = (255,255,255)

def textDelete(event, x,y, flags, param):
    global ix, iy, drawing, img, origin, back, mask, maskBack, maskTemp, mode, rad, color

    trace = []


    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'RECT':
            ix = x
            iy = y
        elif mode == 'DRAW':
            maskTemp = np.zeros(mask.shape,np.uint8)
            cv2.circle(img,(x,y),rad,color,-1)
            cv2.circle(maskTemp,(x,y),rad,(0,0,255),-1)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == 'RECT':
            if drawing == True:
                img = origin.copy()
                cv2.rectangle(img,(ix,iy),(x,y),(255,0,0),1)
        elif mode == 'DRAW':
            if drawing == True:
                cv2.circle(img,(x,y),rad,color,-1)
                cv2.circle(maskTemp,(x,y),rad,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        if mode == 'RECT':
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
        elif mode == 'DRAW':
            if color[0] > 128:
                imgTemp = cv2.bitwise_not(origin)
            else:
                imgTemp = origin.copy()
            maskBack = mask.copy()
            back = origin.copy()
            origin = img.copy()
            maskTemp = cv2.bitwise_and(imgTemp,maskTemp)
            mask = cv2.add(mask,maskTemp)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == 'DRAW':
            color = (img[y,x]).tolist()
            print('mode DRAW  rad = '+str(rad)+" color is ",color)






def main(srcpath,dstpath) :
    global img, origin, back, mask, maskBack, drawing, mode, rad , color
    Image=cv2.imread(srcpath,cv2.IMREAD_COLOR)
    Mask = np.zeros(Image.shape,np.uint8)
    roiNum = 1
    reset = False
    color = (255,255,255)
    mode = 'RECT'
    print('mode RECT')
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
            elif k == 113:
                rad = 10
                mode = 'RECT'
                return -1
            elif k == 114:
                cv2.destroyAllWindows()
                reset = True
                rad = 10
                mode = 'RECT'
                break
            elif k == 109 and drawing == False:
                if mode == 'RECT':
                    mode = 'DRAW'
                    print('mode DRAW  rad = '+str(rad)+" color is ",color )
                elif mode == 'DRAW':
                    mode = 'RECT'
                    print('mode RECT')
            elif k == 43 and drawing == False and mode == 'DRAW':
                if rad < 30:
                    rad += 1
                    print('mode DRAW  rad = '+str(rad)+" color is ",color)
            elif k == 45 and drawing == False and mode == 'DRAW':
                if rad > 1:
                    rad -= 1
                    print('mode DRAW  rad = '+str(rad)+" color is ",color)




        if reset == True:
            main(srcpath,dstpath)
            return -1
        roiNum += 1

    cleanName = dstpath + "_clean.png"
    maskName = dstpath + "_mask.png"

    cv2.imwrite(cleanName,Image)
    cv2.imwrite(maskName,Mask)




if __name__ == "__main__" :
    main(sys.argv[1])