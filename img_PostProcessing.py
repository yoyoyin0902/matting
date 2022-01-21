import cv2
import numpy as np
import skimage.morphology as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io,color

savefile_path = "/home/user/matting/preprocessingfile/"
# savefile_path = "/home/user/matting/preprocessingfile/com/"

def opening(image_path,com_path,fgr_path):
    img = cv2.imread(image_path)
    cv2.imshow('img',img)
    com_img = cv2.imread(com_path)
    print(type(com_img))
    print(type(img))
    print(img.shape, com_img.shape)
    fgr=cv2.imread(fgr_path)
    cv2.imshow('fgr',fgr)
    lunkuo = cv2.bitwise_xor(img,com_img, dst=None, mask=None) #设置80为低阈值，255为高阈值 #輪廓
    cv2.imshow('lunkuo',lunkuo)

    ret,binary=cv2.threshold(lunkuo,140,255,cv2.THRESH_BINARY) #二值化
    cv2.imshow('binary',binary)

    

    kernel_erode= np.ones((5,5),np.uint8) #侵蝕
    kernel_dilation= np.ones((5,5),np.uint8) #膨脹

    img = cv2.erode(img,kernel_erode,iterations =5)
    img = cv2.dilate(img,kernel_dilation,iterations = 6)
    #cv2.imshow('kernel_dilation',img)
    # plt.imshow(img)
    # plt.show()

    
    #plt.show()

    #cv2.imshow("erode",img)
    #cv2.imwrite(savefile_path+"dilate.jpg",img)

    #ret,binary=cv2.threshold(img,140,255,cv2.THRESH_BINARY) #二值化
    
    # contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    # img = cv2.drawContours(img.copy(),contours,1,(0,0,255),3) 
    #cv2.imshow('binary',binary)

    #ret, thresh = cv2.threshold(binary, 127, 255, 0)
    #contours,hierarchy = cv2.threshold(binary, 127, 255, 0)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("bitwise_not",hierarchy)
    #aa=cv2.drawContours(binary, contours, -1, (0,255,255), 3)
    #lunkuo = cv2.bitwise_and(img,hierarchy) #设置80为低阈值，255为高阈值 #輪廓
    #cv2.imshow("bitwise_not",aa)
    


    #cv2.imwrite(savefile_path+"binary.jpg",binary)

    # cv2.imshow("img",img)

    

    # erosion = cv2.erode(img,kernel,iterations = 1)
    # cv2.imshow('erosion',erosion)
    # ret,binary=cv2.threshold(img,140,255,cv2.THRESH_BINARY)
    # cv2.imshow('binary',binary)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def Canny(image_path):




if __name__ == '__main__':
    img_path = "/home/user/matting/img_output/pha/pha2.jpg"
    com_path = "/home/user/matting/img_output/com/com2.png"
    fgr_path = "/home/user/matting/img_output/fgr/fgr2.jpg"
    #opening(img_path)
    opening(img_path,com_path,fgr_path)
