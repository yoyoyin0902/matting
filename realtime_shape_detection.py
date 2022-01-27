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

def empty(a):
    pass



def takePhoto():
    img_savefile = "/home/user/matting/imagedata/img/"

    cam = cv2.VideoCapture(0)
    cam.set(3,1920)
    cam.set(4,1080)
    if not cam.isOpened() | cam.isOpened():
        print("Cannot open camera")
        exit()
    fps = 60

    i = 0
    while(True):
        ret,frame = cam.read()
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k==ord('s'):
            cv2.imwrite(img_savefile+str(i)+'.jpg',frame)
            i+=1
        cv2.imshow("capture",frame)
    cam.release()
    cam.destroyAllWindows()


def opening(pha_path,com_path,fgr_path):
      
    pha = [ f for f in listdir(pha_path) if isfile(join(pha_path,f))]
    com = [ f for f in listdir(com_path) if isfile(join(com_path,f))]
    fgr = [ f for f in listdir(fgr_path) if isfile(join(fgr_path,f))]

    #排序
    pha.sort(key = lambda i: int(i.rstrip('_pha.jpg')))
    com.sort(key = lambda i: int(i.rstrip('_com.png')))
    fgr.sort(key = lambda i: int(i.rstrip('_fgr.jpg')))
    
    img_count = len(pha)
    
    images_pha = np.empty(img_count, dtype=object) 
    images_com = np.empty(img_count, dtype=object)
    images_fgr = np.empty(img_count, dtype=object)

    imgBlur = np.empty(img_count, dtype=object)
    imgGray = np.empty(img_count, dtype=object)
    binary = np.empty(img_count, dtype=object)
    imgErode = np.empty(img_count, dtype=object)
    imgDil = np.empty(img_count, dtype=object)
    imgCanny = np.empty(img_count, dtype=object)
    imgDil2 = np.empty(img_count, dtype=object)

    test = np.empty(img_count, dtype=object)

    for i in range(0, len(pha)): 
        
        filester = pha[i].split("_")[0]

        images_pha[i] = cv2.imread(join(pha_path,pha[i]))
        images_com[i] = cv2.imread(join(com_path,com[i]))
        images_fgr[i] = cv2.imread(join(fgr_path,fgr[i]))
        #cv2.imwrite(savefile_path +filester+"_img.jpg",images_pha[i])
       
      
       
        
        # kernel = np.ones((5, 5),np.uint8)
        # imgErode[i] = cv2.erode(images_pha[i], kernel, iterations =5)
        # imgDil[i] = cv2.dilate(imgErode[i], kernel, iterations = 5)
        
        imgGray[i] = cv2.cvtColor(images_pha[i], cv2.COLOR_BGR2GRAY)
        imgBlur[i] = cv2.GaussianBlur(imgGray[i], (5, 5), 3)
        #cv2.imwrite(savefile_path +filester+"_imgBlur1.jpg",imgBlur[i])
        
        ret,binary[i]=cv2.threshold(imgBlur[i],80,255,cv2.THRESH_BINARY) #二值化
        #binary[i]  = cv2.adaptiveThreshold(imgBlur[i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,3)
        

        kernel = np.ones((3, 3),np.uint8)
        imgErode[i] = cv2.erode(binary[i], kernel, iterations =3)
        imgDil[i] = cv2.dilate(imgErode[i], kernel, iterations = 4)

        imgCanny[i] = cv2.Canny(imgDil[i],10,255)
        

        kernel = np.ones((5, 5),np.uint8)
        imgDil2[i] = cv2.dilate(imgCanny[i], kernel, iterations=1)
        cv2.imwrite(savefile_path +filester+"_img.jpg",binary[i])

        


        
#     #success, img = cap.read()
#     imgContour = img.copy()




#     threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
#     threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
#     ret,binary=cv2.threshold(imgGray,140,255,cv2.THRESH_BINARY) #二值化
    
#     imgCanny = cv2.Canny(binary,threshold1,threshold2)
    

#     kernel = np.ones((5, 5),np.uint8)
#     imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

#     getContours(imgDil,imgContour)


        
        

        
        
        
        #cv2.dilate(images_pha[i], kernel, iterations=5)
        # cv2.imwrite(savefile_path +filester+"_imgBlur.jpg",imgDil[i])
        #cv2.cvtColor(images_com[i], cv2.COLOR_BGR2GRAY)
        #ret,binary[i]=cv2.threshold(images_pha[i],140,255,cv2.THRESH_BINARY) #二值化
        # test[i] = cv2.bitwise_xor(images_pha[i], images_com[i])

        # cv2.imshow('a',images[i]) 
        # imgBlur[i] = cv2.GaussianBlur(images_pha[i], (11, 11), 10)

        # imgGray[i] = cv2.cvtColor(imgBlur[i], cv2.COLOR_BGR2GRAY)

        # ret,binary[i]=cv2.threshold(imgGray[i],100,255,cv2.THRESH_BINARY) #二值化

        #cv2.imwrite(savefile_path +filester+"_imgBlur.jpg",images_pha[i])


        
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
    #takePhoto() #takePhoto
    dataset_root_path = r"/home/user/matting/img_output/"
    pha_path = os.path.join(dataset_root_path,"pha")
    com_path = os.path.join(dataset_root_path,"com")
    fgr_path = os.path.join(dataset_root_path,"fgr")
    opening(pha_path,com_path,fgr_path)