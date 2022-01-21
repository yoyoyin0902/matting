import cv2
import numpy as np
from PIL import Image
import os
import glob 
from os import listdir
from os.path import isfile, join 


savefile_path = "/home/user/matting/preprocessingfile/"

# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

# def empty(a):
#     pass

# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters",300,300)
# cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
# cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
# cv2.createTrackbar("Area","Parameters",5000,30000,empty)

# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver

# def getContours(img,imgContour):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt) #輪廓面積
#         areaMin = cv2.getTrackbarPos("Area", "Parameters")
#         if area > areaMin:
#             cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 3) #畫出輪廓
#             peri = cv2.arcLength(cnt, True) #輪廓周長
#             approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
#             print(len(approx))
#             #x , y , w, h = cv2.minAreaRect(cnt) #外接矩陣
#             rect = cv2.minAreaRect(cnt)
#             box = np.int0(cv2.boxPoints(rect))
#             cv2.drawContours(imgContour, [box], 0, (255, 251, 0), 5)
#             #cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

#             # cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
#             #             (0, 255, 0), 2)
#             # cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
#             #             (0, 255, 0), 2)

# while True:
#     img_path = "/home/user/matting/img_output/pha/pha2.jpg"
#     img2_path = "/home/user/matting/img_output/com/com2.png"
#     img = cv2.imread(img_path)
#     img2 = cv2.imread(img2_path)


#     #success, img = cap.read()
#     imgContour = img.copy()


#     imgBlur = cv2.GaussianBlur(img, (11, 11), 20)
#     imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

#     threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
#     threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
#     ret,binary=cv2.threshold(imgGray,140,255,cv2.THRESH_BINARY) #二值化
    
#     imgCanny = cv2.Canny(binary,threshold1,threshold2)
    

#     kernel = np.ones((5, 5),np.uint8)
#     imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

#     getContours(imgDil,imgContour)
#     imgStack = stackImages(0.8,([binary,imgCanny],[imgDil,imgContour]))
#     imgStack = stackImages(0.8,([img2,img]))
    
#     cv2.imwrite(savefile_path+"imgStack1.jpg",imgStack)
#     cv2.imshow("Result", imgStack)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

def opening(pha_path,com_path,fgr_path):
    # sorted(os.listdir(pha_path))
    # print(pha_path)
    #count = len(imglist)
    onlyfiles = [ f for f in listdir(pha_path) if isfile(join(pha_path,f)) ]
    
    images = np.empty(len(onlyfiles), dtype=object) 
    imgBlur = np.empty(len(onlyfiles), dtype=object)
    imgGray = np.empty(len(onlyfiles), dtype=object)
    binary = np.empty(len(onlyfiles), dtype=object)


    onlyfiles.sort()
    print(onlyfiles)
    for i in range(0, len(onlyfiles)): 
        
        filester = onlyfiles[i].split("_")[0]
        #print(filester)
        images[i] = cv2.imread(join(pha_path,onlyfiles[i]))
        # cv2.imshow('a',images[i]) 
        imgBlur[i] = cv2.GaussianBlur(images[i], (11, 11), 20)

        imgGray[i] = cv2.cvtColor(imgBlur[i], cv2.COLOR_BGR2GRAY)

        ret,binary[i]=cv2.threshold(imgGray[i],140,255,cv2.THRESH_BINARY) #二值化

        cv2.imwrite(savefile_path +filester+"_imgBlur.jpg",binary[i])


        
        #cv2.destroyAllWindows() 

    
    # for i in range(count):
    #     filester = imglist[i].split(".")[0]
    #     print(filester)
    #     pha_img = pha_path +"/"+ filester +".jpg"
     
    #     pha = cv2.imread(pha_img)
    #     cv2.imshow("Result", pha)

    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        

        




if __name__ == '__main__':
    dataset_root_path = r"/home/user/matting/img_output/"
    pha_path = os.path.join(dataset_root_path,"pha")
    com_path = os.path.join(dataset_root_path,"com")
    fgr_path = os.path.join(dataset_root_path,"fgr")
    opening(pha_path,com_path,fgr_path)