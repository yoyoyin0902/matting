#!/usr/bin/env python3

import cv2
import depthai as dai

img_savefile = "/home/user/matting/imagedata/img/"

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # print('Connected cameras: ', device.qgetConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    i = 0
    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        color_image = inRgb.getCvFrame()
        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("rgb", color_image)

        print(color_image.shape)

        if cv2.waitKey(1)==ord('s'):
            i+=1
            cv2.imwrite(img_savefile+str(i)+'.jpg',color_image)
            

        if cv2.waitKey(1) == ord('q'):
            break

# # savefile_path = "/home/user/matting/preprocessingfile/"
# savefile_path = "/home/user/shape_detection/process/"

# savefile_Left  = "/media/user/Extreme SSD/takeCam/yolo/imgL/"
# savefile_Right = "/media/user/Extreme SSD/takeCam/yolo/imgR/"

# pha_path = "/home/user/shape_detection/pha/long_2.jpg"

# img = cv2.imread(pha_path,0)
# cv2.imshow("img",img)
# imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
# # cv2.imshow("img1",imgBlur)

# # kernel = np.ones((3, 3),np.uint8)
# # imgDil= cv2.dilate(imgBlur, kernel, iterations = 2)
# # imgErode = cv2.erode(imgDil, kernel, iterations = 2)
# # imgDil= cv2.dilate(imgErode, kernel, iterations = 4)
# # # cv2.imshow("binary1",binary1)
# # cv2.imshow("imgDil",imgDil)

# # ret,binary1=cv2.threshold(imgDil,120,255,cv2.THRESH_BINARY) #二值化
# ret,binary2=cv2.threshold(imgBlur,110,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #二值化
# # cv2.imshow("binary1",binary1)
# cv2.imshow("binary2",binary2)


# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def TwoCamera():
#     camR = cv2.VideoCapture(2)
#     camL = cv2.VideoCapture(0)
#     print(camR)

#     camR.set(3,960)
#     camR.set(4,540)

#     camL.set(3,960)
#     camL.set(4,540)
#     fps = 60

#     # if not camR.isOpened():
#     #     print("Cannot open cameraRight")
#     #     exit()
    
#     # if not camL.isOpened():
#     #     print("Cannot open cameraLeft")
#     #     exit()
    
#     i = 0
#     while(True):
#         retR, frameR = camR.read()
#         retL, frameL = camL.read()
#         k = cv2.waitKey(1)
#         if k == 27:
#             break
#         elif k==ord('s'):
#             cv2.imwrite(savefile_Left+str(i)+'.jpg',frameL)
#             cv2.imwrite(savefile_Right+str(i)+'.jpg',frameR)
#             i+=1

#         cv2.imshow("captureR",frameR)
#         cv2.imshow("captureL",frameL)

#     camR.release()
#     camL.release()
#     cv2.destroyAllWindows()
#     # camL.destroyAllWindows()


# def takePhoto():
#     img_savefile = "/home/user/matting/imagedata/img/"

#     pipeline = dai.Pipeline()

#     camRgb = pipeline.create(dai.node.ColorCamera) #cam

#     xoutRgb = pipeline.create(dai.node.XLinkOut)

#     xoutRgb.setStreamName("rgb")

#     #cam參數調整
#     camRgb.setPreviewSize(1280, 720)
#     # camRgb.setIspScale(2,3)
#     # camRgb.initialControl.setManualFocus(135)
#     camRgb.setBoardSocket(dai.CameraBoardSocket.RGB) #要使用的相機
#     camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) #分辨率 1080p 12mp q
#     camRgb.setFps(30)
#     camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB) #输出图像的颜色顺序，RGB或BGR
#     camRgb.setInterleaved(False) #设置输出图像是否交错数据

#     camRgb.preview.link(xoutRgb.input)

#     with dai.Device(pipeline) as device:
#         device.startPipeline()
#         qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
#         imageIn = qRgb.get()

#         color_image = imageIn.getCvFrame()
#         print(color_image.shape)

#         i = 0
#         while(True):
#             # ret,frame = cam.read()
#             k = cv2.waitKey(1)
#             if k == 27:
#                 break
#             elif k==ord('s'):
#                 cv2.imwrite(img_savefile+str(i)+'.jpg',color_image)
#                 i+=1
#             elif k ==ord('q'):
#                 break

#             cv2.imshow("color_image",color_image)

#     # cam.release()
#     cv2.destroyAllWindows()

# def calcAndDrawHist(image, color):
#     hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
#     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
#     histImg = np.zeros([256, 256, 3], np.uint8)
#     hpt = int(0.9*256)

#     for h in range(256):
#         intensity = int(hist[h] * hpt / maxVal)
#         cv2.line(histImg, (h, 256), (h, 256-intensity), color)

#     return histImg

# def show_histphoto(photo_path):
#     image = cv2.imread(photo_path)
#     b, g, r = cv2.split(image)

#     histImgB = calcAndDrawHist(b, [255, 0, 0])
#     histImgG = calcAndDrawHist(b, [0, 255, 0])
#     histImgR = calcAndDrawHist(b, [0, 0, 255])

#     cv2.imshow('histImgB', histImgB)
#     cv2.imshow('histImgG', histImgG)
#     cv2.imshow('histImgR', histImgR)
#     # cv2.imshow('Img', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def drawplot():
#     plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
#     y_maxValue = np.max(grayHist)
#     plt.axis([0, 255, 0, y_maxValue]) # x和y的範圍
#     plt.xlabel("gray Level")
#     plt.ylabel("Number Of Pixels")
#     plt.show()


# def Line_chart(photo_path):
    
#     image = cv2.imread(photo_path)

#     colors = ("b","g","r")

#     plt.figure()
#     plt.title("Flattened Color Histogram")
#     plt.xlabel("Bins")
#     plt.ylabel("# of Pixels")

#     for i,col in enumerate(colors):
#         hist = cv2.calcHist([image],[i],None,[256],[0,256])
#         plt.plot(hist,color = col)
#         plt.xlim([0,256])
#     plt.show()
#     cv2.imwrite(savefile_path+"histogram.jpg",clahe_img)
    

# def equalization(img_path):
#     img = cv2.imread(img_path)
#     # cv2.imshow("original",img_path)

#     green = img[:,:,0]
#     blue = img[:,:,1]
#     red = img[:,:,2]
#     #equalization
#     red_EDU = cv2.equalizeHist(red)
#     blue_EDU = cv2.equalizeHist(blue)
#     green_EDU = cv2.equalizeHist(green)

#     img_equal = img.copy()

#     img_equal[:,:,0] = green_EDU
#     img_equal[:,:,1] = blue_EDU
#     img_equal[:,:,2] = red_EDU
#     # print(img_equal)

#     # cv2.imshow("after", img_equal)
#     cv2.imwrite(savefile_path+"equal.jpg",img_equal)

#     clahe = cv2.createCLAHE()
#     clahe_img = clahe.apply(img_equal)

#     cv2.imwrite(savefile_path+"clahe.jpg",clahe_img)

#     plt.show()
#     cv2.waitKey(0)
#     #cv2.destroyAllWindows()


#     return img_equal

# def opening(pha_pat):
      
#     pha = [ f for f in listdir(pha_path) if isfile(join(pha_path,f))]
#     # com = [ f for f in listdir(com_path) if isfile(join(com_path,f))]
#     # fgr = [ f for f in listdir(fgr_path) if isfile(join(fgr_path,f))]

#     #排序
#     # pha.sort(key = lambda i: int(i.rstrip('_pha.jpg')))
#     # com.sort(key = lambda i: int(i.rstrip('_com.png')))
#     # fgr.sort(key = lambda i: int(i.rstrip('_fgr.jpg')))
    
#     img_count = len(pha)
    
#     images_pha = np.empty(img_count, dtype=object) 
#     # images_com = np.empty(img_count, dtype=object)
#     # images_fgr = np.empty(img_count, dtype=object)

#     gray_img = np.empty(img_count, dtype=object)
#     imgBlur = np.empty(img_count, dtype=object)
#     bin_img = np.empty(img_count, dtype=object)
#     imgErode = np.empty(img_count,dtype=object)
#     imgDil = np.empty(img_count,dtype=object)
#     hierarchy = np.empty(img_count,dtype=object)
#     ext = np.empty(img_count,dtype=object)
#     binary = np.empty(img_count,dtype=object)
    

#     test = np.empty(img_count, dtype=object)

#     for i in range(0, len(pha)): 
        
#         # filester = pha[i].split("_")[0]

#         images_pha[i] = cv2.imread(join(pha_path,pha[i]))
#         # images_com[i] = cv2.imread(join(com_path,com[i]))
#         # images_fgr[i] = cv2.imread(join(fgr_path,fgr[i]))
        
        
#         imgBlur[i] = cv2.GaussianBlur(images_pha[i], (5, 5), 0)
#         gray_img[i] = cv2.cvtColor(imgBlur[i],cv2.COLOR_BGR2GRAY)

#         ret,binary[i]=cv2.threshold(gray_img[i],50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #二值化
        
#         kernel = np.ones((5, 5),np.uint8)
#         imgErode[i] = cv2.erode(binary[i], kernel, iterations = 4)
#         imgDil[i] = cv2.dilate(imgErode[i], kernel, iterations = 4)

#         # ret,bin_img[i] = cv2.threshold(imgDil[i],80,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         # # contours, hierarchy[i] = cv2.findContours(bin_img[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         # # ext[i] = cv2.drawContours(imgBlur[i],contours,-1,(255,255,255),3)

        
#         cv2.imwrite(savefile_path +str(i)+"_old.jpg",images_pha[i])
#         cv2.imwrite(savefile_path +str(i)+"_img.jpg",imgDil[i])
        

#         # contours, hierarchy[i]	=cv.findContours(images_pha[i],mode= RETR_LIST,method = CHAIN_APPROX_SIMPLE
#         #     ,contours[, hierarchy[, offset]]])
       

       
        
                 


# if __name__ == '__main__':
#     takePhoto() #takePhot
#     # TwoCamera()
#     #dataset_root_path = r"/home/user/shape_detection/pha/"
#     pha_path = r"/home/user/shape_detection/pha/"
#     # com_path = os.path.join(dataset_root_path,"com")
#     # fgr_path = os.path.join(dataset_root_path,"fgr")
#     opening(pha_path)


   
    
