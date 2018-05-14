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
import ballTextFinderCont

ix, iy = -1,-1
drawing = False
img = []
origin = []
back = []

def textDelete(event, x,y, flags, param):
    global ix, iy, drawing, img, origin, back

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
        textFinder= ballTextFinderCont.TextFinder()
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
        img[y1:y2+1,x1:x2+1] = textFinder.FindText(img[y1:y2+1,x1:x2+1])
        back = origin.copy()
        origin = img.copy()
        #cv2.rectangle(img,(ix,iy),(x,y),(255,0,0),-1)
        #print(x-ix,y-iy)





def main(string) :
    global img, origin, back
    Image=cv2.imread(string,cv2.IMREAD_COLOR)
    roiNum = 1
    while True:
        if Image.shape[0] <= 720*(roiNum-1) - 100:
            break
        elif Image.shape[0] > 720*roiNum - 100:
            if roiNum == 1:
                img = Image[720*(roiNum-1):720*roiNum+1- 100]
            else:
                img = Image[720*(roiNum-1)-100:720*roiNum+1- 100]
        else:
            img = Image[720*(roiNum-1)-100:]

        origin = img.copy()
        back = origin.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',textDelete)

        while True:
            cv2.imshow('image',img)
            k = cv2.waitKey(1)

            if k == 27:
                cv2.destroyAllWindows()
                if Image.shape[0] > 720*roiNum:
                    if roiNum == 1:
                        Image[720*(roiNum-1):720*roiNum+1- 100] = origin
                    else:
                        Image[720*(roiNum-1)-100:720*roiNum+1- 100] = origin
                else:
                    Image[720*(roiNum-1)-100:] = origin
                break
            elif k == 26:
                origin = back.copy()
                img = origin.copy()

        roiNum += 1

    outputName = "clean" + string
    cv2.imwrite(outputName,Image)


if __name__ == "__main__" :
    main(sys.argv[1])
