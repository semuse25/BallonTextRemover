import cv2
import numpy as np
from matplotlib import pyplot as plt

def deleteCanny(canny, cnt):
    img_h, img_w = canny.shape

    for i in cnt:
        canny[i[0,1],i[0,0]] = 0

class BalloonCleaner:
    def insidePoint(self, img):
        h,w = img.shape[:2]

        preColor = img[h//2,w//2]
        num = 0
        first = 0
        second = 0
        for i in range(w//2):
            if (preColor == [0,0,0]).all() and (img[h//2,i] == [255,255,255]).all():
                num += 1
                if first == 0:
                    first = i
                elif second == 0:
                    second = i
            preColor = img[h//2,i]
        if num % 2 == 1:
            return [h//2,w//2]

        preColor = img[h//2,w//2]
        num = 0
        first = 0
        second = 0
        for i in range(w//2,w):
            if (preColor == [0,0,0]).all() and (img[h//2,i] == [255,255,255]).all():
                num += 1
                #cv2.circle(img,(i,h//2),5,(255,0,0),1)
                if first == 0:
                    first = i
                elif second == 0:
                    second = i
            preColor = img[h//2,i]

        #print(num)
        if num % 2 == 1:
            #print("a")
            return [h//2,w//2]
        elif num == 0:
            first, second = 0,0
            for i in range(1,w//2-5):
                if (img[h//2,w//2-i] == [255,255,255]).any():
                    if first == 0:
                        first = w//2 - i
                    elif second == 0:
                        second = w//2 - i
                        return [h//2,(first+second)//2]
        else:
            return [h//2,(first+second)//2]


    def cleanBalloon(self, image):
        #img = cv2.imread('ball9.jpg')
        img = image.copy()
        blackMargin = 5
        img_h,img_w = img.shape[:2]
        frame = np.zeros((img_h+(2*blackMargin),img_w+(2*blackMargin),3), np.uint8)
        mask = np.zeros((img_h+(2*blackMargin),img_w+(2*blackMargin),3), np.uint8)
        frame[blackMargin:img_h+blackMargin,blackMargin:img_w+blackMargin] = img

        bin = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(bin,220,255,cv2.THRESH_BINARY)
        ##for i in bin:
        ##    if np.count_nonzero(i) > 0:
        ##        i = [255,255,255]
        ##    else:
        ##        i = [0,0,0]

        # converting to gray scale
        #gray = cv2.cvtColor(bin, cv2.COLOR_BGR2GRAY)

        # remove noise
        img = cv2.GaussianBlur(bin,(3,3),0)

        # convolute with proper kernels
        laplacian = cv2.Laplacian(img,cv2.CV_8U)
        #laplacian=cv2.Canny(laplacian,150,300)

        #r, laplacian = cv2.threshold(laplacian,1,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        #laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, kernel, iterations=1)
        laplacian = cv2.dilate(laplacian,kernel,iterations = 1)

        cnts,contours,hierarchy  = cv2.findContours(laplacian, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print("none!")

        for i in range(len(contours)):
            cnt=contours[i]
            if cv2.arcLength(cnt, True) < (img_h + img_w):
                deleteCanny(laplacian,cnt)

        #laplacian=cv2.Canny(laplacian,150,300)
        ##
        ##print(img.shape)
        ##print(laplacian.shape)
        ##print(canny.shape)
        ##
        ##print(type(laplacian))


        #cv2.imshow("lap",laplacian)
        #cv2.waitKey(0)



        cnts,contours,hierarchy  = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = img_h*img_w
        n = 0
        hie = hierarchy[0]
        for i in range(len(hie)):
            if cv2.contourArea(contours[i]) > (img_h * img_w)*0.25:
                if cv2.contourArea(contours[i]) < area:
                   area = cv2.contourArea(contours[i])
                   n = i


        #cv2.drawContours(frame,contours[n],-1,(0,0,255),-1)

        #cv2.drawContours(frame, contours[n], -1, (255,255,255), -1)
        #cv2.drawContours(frame, contours[n], -1, (0,0,0), 3)
        #print(contours[n][:,0][0])
        #point = (int(contours[n][:,0][:,0].mean()),int(contours[n][:,0][:,1].mean()))

        #point = contours[n][:,0][0]
        mask2 = mask.copy()
        cv2.drawContours(mask, contours[n], -1, (255,255,255), 2)
        #cv2.imshow("mask",mask)
        #cv2.waitKey(0)
        #binMask = cv2.threshold(mask,220,255,cv2.THRESH_BINARY)[1]
        grayMask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("mask",binMask)
        #cv2.waitKey(0)
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

        #point = self.insidePoint(mask)
        #if point == None:
            #return img
        #print(point)
        #cv2.circle(mask,(point[1],point[0]),5,(255,255,255),1)
        #cv2.circle(mask,(mask.shape[1]//2,mask.shape[0]//2),5,(255,255,255),1)
        #cv2.imshow("maskTemp",mask)
        #cv2.waitKey(0)
        floodflags = 8
        floodflags |= (255 << 8)
        floodMask = np.zeros((img_h+(2*blackMargin)+2,img_w+(2*blackMargin)+2), np.uint8)
        cv2.floodFill(mask, floodMask, (0,0), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (0,img_h+(2*blackMargin)-1), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (img_w+(2*blackMargin)-1,0), (255,255,255),(0,0,0),(0,0,0),floodflags)
        cv2.floodFill(mask, floodMask, (img_w+(2*blackMargin)-1,img_h+(2*blackMargin)-1), (255,255,255),(0,0,0),(0,0,0),floodflags)
        #floodMask = cv2.bitwise_not(floodMask)
        mask = cv2.bitwise_not(mask)

        #mask = floodMask[1:img_h+(2*blackMargin)+1,1:img_w+(2*blackMargin)+1]
        #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        #cv2.drawContours(mask, contours[n], -1, (0,0,0), 3)

        #cv2.imshow("mask",mask)
        #cv2.waitKey(0)
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

        ##    for i in range(len(contours)):
        ##        cnt=contours[i]
        ##        if area < cv2.arcLength(cnt, True):
        ##            area = cv2.arcLength(cnt, True)
        ##            contour[0] = cnt
        ##
        ##    if area < (img_h+img_w)/5:
        ##        laplacian = cv2.dilate(laplacian,kernel,iterations = 1)
        ##    else:
        ##        break

        ##    for i in range(len(contours)):
        ##        cnt=contours[i]
        ##        if area < cv2.contourArea(cnt) and cv2.contourArea(cnt) <= img_h*img_w*0.95:
        ##            area = cv2.contourArea(cnt)
        ##            contour[0] = cnt
        ##
        ##    if area < img_h*img_w//5:
        ##        laplacian = cv2.dilate(laplacian,kernel,iterations = 1)
        ##    else:
        ##        break

        ##contPoints = contour[0][:,0]
        ##point = (int(contPoints[:,0].mean()),int(contPoints[:,1].mean()))
        ##
        ##
        ##print(point)
        ##num = 0
        ##near = 0
        ##for i in contPoints:
        ##    if point[0] == i[0]:
        ##        if point[1] > i[1]:
        ##            num += 1
        ##            if i[1] > near:
        ##                near = i[1]
        ##
        ##print(num)
        ##
        ##if (bin[point[1],point[0]] == (0,0,0)).all():
        ##    for i in range(1, point[1]-near):
        ##        if (bin[point[1]-i,point[0]] == (255,255,255)).all():
        ##            point = (point[0], point[1] - i)
        ##            break
        ##
        ##
        ##if (bin[point[1],point[0]] == (0,0,0)).all():
        ##    print("black")
        ##else:
        ##    print("white")

        ##floodflags = 8
        ##floodflags |= (255 << 8)
        ##mask = np.zeros((img_h+(2*blackMargin)+2,img_w+(2*blackMargin)+2), np.uint8)
        ##im_floodfill = laplacian.copy()
        ##cv2.floodFill(im_floodfill, mask, (point[0]+1,point[1]+1), (255,255,255),(0,0,0),(0,0,0),floodflags)
        ##mask_inv = cv2.bitwise_not(mask)

        ##im_floodfill = bin.copy()
        ##cv2.floodFill(im_floodfill, mask, (point[0]+1,point[1]+1), 128,(0,0,0),(0,0,0),floodflags)
        ##im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        ##binv = cv2.bitwise_not(bin)
        ##im_out = binv | im_floodfill_inv
        ##th, im_th2 = cv2.threshold(im_out, 130, 255, cv2.THRESH_BINARY)
        ##im_th3 = cv2.bitwise_not(im_th2)
        ##cnts,contour,hier = cv2.findContours(im_th3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        ##
        ##for cnt in contour:
        ##    cv2.drawContours(im_th3,[cnt],0,255,-1)
        ##
        ##segm = cv2.bitwise_not(im_th3)
        ##segm_inv = cv2.bitwise_not(segm)
        ##
        ##frame = cv2.bitwise_and(frame, frame, mask=segm_inv)


        ##cv2.drawContours(mask, contour, -1, (255,255,255), -1)
        ##cv2.drawContours(mask, contour, -1, (0,0,0), 3)
        ##hImg = cv2.bitwise_and(frame,mask)
        ##gray = cv2.cvtColor(hImg,cv2.COLOR_BGR2GRAY)
        ##hist = cv2.calcHist([gray],[0],None,[256],[1,256])
        ##hist = hist.tolist()
        ##colMax = max(hist)
        ##bgColor = hist.index(colMax)


