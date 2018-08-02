# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import imagetool

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,3))

def bubbleChecker(ori,img,i,x,y,w,h):
    #1. blob size limit-------------------

    rowMin = int(img.shape[1] / 35)  #minimal size limit
    colMin = int(img.shape[0] / 35)
    rowMax = int(img.shape[1] / 1.5)
    colMax = int(img.shape[0] / 1.5)

    if (w < rowMin) or (h < colMin) :
        print('%d is deleted by size'%i)
        return 0

    if (w > rowMax) or (h > colMax) :
        print('%d is deleted by size!' % i)
        return 0

    if h * 1.5 < w :
        print('%d is deleted by sizerate' % i)
        return 0

    #2. white pixel rate limit----------------------
    img_trim = img[y:y + h, x:x + w]

    n_white_pix = cv2.countNonZero(img_trim)
    whiterate =(  n_white_pix / (w*h) ) * 100
    if whiterate < 45 :
        print('%d is deleted by whiterate' % i)
        return 0

    #3. two line
    edges = cv2.Canny(img_trim, 0, 0, apertureSize=3)

    if h < int(img.shape[0] / 5) :
        minLineLength = (h * 19) / 100
    else:
        minLineLength = (h * 30) / 100
    maxLineGap = w/5
    lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi / 500, threshold=20,
                            lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

    vcount = 0

    # For DEBUG --------------------------------------------
    if i==-1:
        print(h*w)
        print((h*w )/500)
        if lines is not None :
            for i in range(len(lines)):
                for x1, y1, x2, y2 in lines[i]:
                    if (x1-x2) == 0 :
                        vcount +=1
                    cv2.line(img_trim, (x1, y1), (x2, y2), (0, 0, 255), 3)


        print('vertical line : %d'%vcount)
        cv2.imshow('%d'%i, img_trim)
        cv2.waitKey(0)
    #-----------------------------------------------------------


    if lines is None:
        print ('No.%d is modified'%i)
        img_trim = cv2.erode(img_trim, kernel, iterations=1)
        edges = cv2.Canny(img_trim, 0, 0, apertureSize=3)
        lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi / 500, threshold=20,
                                lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is None:
            print(0)
            print('%d is deleted by line' % i)
            return 0


    a, b, c = lines.shape
    hcount = 0
    vcount = 0
    dcount =0

    for n in range(a):
        x1 = lines[n][0][0]
        x2 = lines[n][0][2]
        y1 = lines[n][0][1]
        y2 = lines[n][0][3]
        if (x2 - x1) == 0:
            vcount += 1
        if (y2 - y1) == 0:
            hcount += 1
        if ((x2 - x1) != 0 ) and ((y2 - y1) != 0 ) :
            dcount+=1


    if (vcount == 0 )or 1< vcount < 2:
        print('%d is deleted by line' % i)
        return 0

    if vcount > (h*w )/300  :
        print('%d is deleted by TOO MANY vertical line:%d' % (i,vcount))
        return 0

    if dcount >= (vcount/8) :
        print('%d is deleted by diagonal line:%d' % (i, dcount))
        return 0

    ori_trim = ori[y:y + h, x:x + w]
    bcount = imagetool.blobDetect(ori_trim)
    ccount = imagetool.connectedComponentDetect(img_trim)
    if bcount == 0:
        print('%d is deleted by blob detection ... 0')
        return 0

    std = np.std(ori_trim.ravel())

    if std > 99:
        print('%d is deleted by histogram STD %d'%(i,std))
        return 0

    '''
    forCNN = imagetool.imgResizer(ori_trim)
    if test(forCNN) is False:
        print('%d is deleted by CNN'%i)
        return 0
    '''

    print('---------%d is selected! v:%d h:%d d:%d blob:%d STD:%d CC:%d----------' %(i,vcount,hcount,dcount,bcount,std,ccount))

    return 1



def bubbleFinder(image):
    # load the image, convert it to grayscale, and blur it

    image2 = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 224, 250, cv2.THRESH_BINARY)[1]

    # threshold the image to reveal light regions in the
    # blurred image

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh,kernel,iterations = 1)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=4)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):

        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"

        if numPixels > 2000:
            mask = cv2.add(mask, labelMask)

    data=[]
    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    if len(cnts) == 0 :
        return data

    cnts = contours.sort_contours(cnts)[0]

    predata=[]
    class Cnt:
        def __init__(self, x, y, w, h, area):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.area = area

        def __repr__(self):
            return repr((self.x, self.y, self.w,self.h,self.area))

    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        tmp =Cnt(x,y,w,h,w*h)
        predata.append(tmp)

    sorted_pre = sorted(predata, key=lambda Cnt: Cnt.area)

    i=0
    for cnts in sorted_pre:
        x=cnts.x
        y=cnts.y
        w=cnts.w
        h=cnts.h
        #cv2.rectangle(image, (x, y), (x + w, y + h), (30, 0, 255), 3)
        #cv2.putText(image, "#{}".format(i), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        print(i, x, y, w, h)
        if bubbleChecker(gray, thresh, i, x, y, w, h) == 1:
            data.append([x, y, w, h])

            #cv2.rectangle(image, (x, y), (x + w, y + h), (30, 0, 255), 3)
            #cv2.putText(image, "#{}".format(i), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        i+=1

    #for [x,y,w,h] in data:
        #cv2.rectangle(image2, (x, y), (x + w, y + h), (30, 0, 255), 3)
        #cv2.putText(image2, "#{}".format(i), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



    return data