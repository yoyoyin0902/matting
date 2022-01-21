import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

savefile_path = "/home/user/matting/preprocessingfile/"


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

def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9*256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256-intensity), color)

    return histImg

def show_histphoto(photo_path):
    image = cv2.imread(photo_path)
    b, g, r = cv2.split(image)

    histImgB = calcAndDrawHist(b, [255, 0, 0])
    histImgG = calcAndDrawHist(b, [0, 255, 0])
    histImgR = calcAndDrawHist(b, [0, 0, 255])

    cv2.imshow('histImgB', histImgB)
    cv2.imshow('histImgG', histImgG)
    cv2.imshow('histImgR', histImgR)
    # cv2.imshow('Img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawplot():
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的範圍
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()


def Line_chart(photo_path):
    
    image = cv2.imread(photo_path)

    colors = ("b","g","r")

    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for i,col in enumerate(colors):
        hist = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    plt.show()
    cv2.imwrite(savefile_path+"histogram.jpg",clahe_img)
    

def equalization(img_path):
    img = cv2.imread(img_path)
    # cv2.imshow("original",img_path)

    green = img[:,:,0]
    blue = img[:,:,1]
    red = img[:,:,2]
    #equalization
    red_EDU = cv2.equalizeHist(red)
    blue_EDU = cv2.equalizeHist(blue)
    green_EDU = cv2.equalizeHist(green)

    img_equal = img.copy()

    img_equal[:,:,0] = green_EDU
    img_equal[:,:,1] = blue_EDU
    img_equal[:,:,2] = red_EDU
    # print(img_equal)

    # cv2.imshow("after", img_equal)
    cv2.imwrite(savefile_path+"equal.jpg",img_equal)

    clahe = cv2.createCLAHE()
    clahe_img = clahe.apply(img_equal)

    cv2.imwrite(savefile_path+"clahe.jpg",clahe_img)

    plt.show()
    cv2.waitKey(0)
    #cv2.destroyAllWindows()


    return img_equal


if __name__ == '__main__':
    takePhoto() #takePhoto
    # photo_path = '/home/user/matting/imagedata/bgr/2.jpg'
    # Line_chart(photo_path)
    #equalization(photo_path)
    #Line_chart("/home/user/matting/equal_bgr.jpg")


   
    
