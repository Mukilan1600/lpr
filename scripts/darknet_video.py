from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import pytesseract
from skimage import measure
import threading
from scipy.spatial import distance as dist
from collections import OrderedDict
from multiprocessing import Process, Lock

lic_pl = cv2.imread("test.png")
f=False

class CentroidTracker:
    def __init__(self,maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID+=1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] >= self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects),2), dtype="int")

        for (i, (cX,cY)) in enumerate(rects):
            inputCentroids[i] = (cX,cY)

        if len(self.objects)==0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row,col) in zip(rows,cols):

                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0,D.shape[0])).difference(usedRows)
            unusedCols = set(range(0,D.shape[1])).difference(usedCols)

            if D.shape[0]>=D.shape[1]:
                for row in  unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def ocr():
    global lic_pl, f
    while(True):
        if(f==True):
            configuration = ("-l eng --oem 1 --psm 8")
            text = pytesseract.image_to_string(lic_pl, config=configuration)
            print(text)
            cv2.imshow("d",lic_pl)
            cv2.waitKey(3)
            f=False

def cvDrawBoxes(detections, img):
    global lic_pl, f
    #img = cv2.resize(img,(1920,1080),interpolation = cv2.INTER_AREA)
    for detection in detections:
        if detection[0]==b'PLATE':
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            #print(x,y)
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (int((xmin/416.0)*img.shape[1]), int((ymin/416.0)*img.shape[0]))
            pt2 = (int((xmax/416.0)*img.shape[1]), int((ymax/416.0)*img.shape[0]))
            #pt1 = (xmin, ymin)
            #pt2 = (xmax,ymax)
            #print(pt1, pt2)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            org_img = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(hsv,(5,5),0)
            ret3,binary_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            c_img = cv2.bitwise_not(binary_img)
            #cv2.imshow("tes",binary_img)
            #cv2.waitKey(0)
            image, contours, hier = cv2.findContours(c_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            xmin=30
            character_dimensions = (0.25*org_img.shape[0], 0.70*org_img.shape[0], 0.01*org_img.shape[1], 0.25*org_img.shape[1])
            min_height, max_height, min_width, max_width = character_dimensions
            new_im = cv2.imread("test.png")
            d=0
            for ctr in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)
                # Getting ROI
                if w>min_width and w<max_width and h>min_height and h<max_height:
                    d+=1
                    roi = cv2.cvtColor(binary_img[y:y+h, x:x+w],cv2.COLOR_GRAY2RGB)
                    #char.append(cv2.resize(roi,(50,75),interpolation = cv2.INTER_AREA))
                    new_im[38:113, xmin:xmin+50] = cv2.resize(roi,(50,75),interpolation = cv2.INTER_AREA)
                    xmin+=70
                    #cv2.imshow('character',roi)
                    #cv2.imwrite('character_%d.png'%d, roi)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
            cv2.imshow("d",new_im[:,:xmin])
            cv2.waitKey(3)
            if d>6 and d<11:
                lic_pl = new_im[:,:xmin]
                configuration = ("-l eng --oem 1 --psm 8")
                text = pytesseract.image_to_string(new_im[:,:xmin], config=configuration)
                print(text)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "yolov3test.cfg"
    weightPath = "yolov3test_last.weights"
    metaPath = "obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("tt.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if ret==False:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        #print(detections)
        image = cvDrawBoxes(detections, frame_rgb)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        #cv2.imshow('Demo', image)
        #cv2.waitKey(3)
    #cv2.imwrite('Demo.png', image)
    #cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
	p = Process(target=YOLO)
	p.start()
