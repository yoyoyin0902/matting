import cv2
import numpy as np
import skimage.morphology as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io,color

savefile_path = "/home/user/matting/preprocessingfile/"
# savefile_path = "/home/user/matting/preprocessingfile/com/"

def opening(image_path,com_path):
    img = cv2.imread(image_path,0)
    com_img = cv2.imread(com_path,0)
    #cv2.imshow('origin',img)
    
    # laplacian = cv2.Laplacian(img,cv2.CV_64F)
    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7)

    # abs_sobelx = np.absolute(sobelx)
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # thresh_min = 10
    # thresh_max = 170
    # sxbinary = np.zeros_like(scaled_sobel)
    # sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # plt.imshow(sxbinary, cmap='gray') 
    # plt.show()  

    # plt.subplot(2,2,1),plt.imshow(img,cmap='jet')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(laplacian,cmap='jet')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobelx,cmap='jet')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4),plt.imshow(sobely,cmap='jet')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # plt.imshow(img,cmap='jet')
    # plt.show()
    # cv2.imwrite(savefile_path+"old1.png",img)

    # kernel = np.ones((20,20),np.uint8)
    # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow('erosion',img)

    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray_img)
    # plt.show()
    #cv2.imshow('gray',gray_img)

    

    kernel_erode= np.ones((5,5),np.uint8) #侵蝕
    kernel_dilation= np.ones((5,5),np.uint8) #膨脹

    img = cv2.erode(img,kernel_erode,iterations =5)
    img = cv2.dilate(img,kernel_dilation,iterations = 6)
    cv2.imshow('kernel_dilation',img)
    # plt.imshow(img)
    # plt.show()

    
    #plt.show()

    #cv2.imshow("erode",img)
    #cv2.imwrite(savefile_path+"dilate.jpg",img)

    ret,binary=cv2.threshold(img,140,255,cv2.THRESH_BINARY) #二值化
    
    # contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    # img = cv2.drawContours(img.copy(),contours,1,(0,0,255),3) 
    cv2.imshow('binary',binary)

    ret, thresh = cv2.threshold(binary, 127, 255, 0)
    #contours,hierarchy = cv2.threshold(binary, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("bitwise_not",hierarchy)
    aa=cv2.drawContours(binary, contours, -1, (0,255,255), 3)
    # lunkuo = cv2.bitwise_or(img,hierarchy) #设置80为低阈值，255为高阈值 #輪廓
    cv2.imshow("bitwise_not",aa)
    


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
    #opening(img_path)
    opening(img_path,com_path)
