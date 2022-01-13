import cv2
import numpy as np
import skimage.morphology as sm
from skimage import io,color

savefile_path = "/home/user/matting/preprocessingfile/"

def opening(image_path):
    img = cv2.imread(image_path)
    # cv2.imshow('old',img)
    cv2.imwrite(savefile_path+"old.jpg",img)
    

    kernel_erode= np.ones((5,5),np.uint8)
    kernel_dilation= np.ones((5,5),np.uint8)

    img = cv2.erode(img,kernel_erode,iterations = 5)
    img = cv2.dilate(img,kernel_dilation,iterations = 5)
    cv2.imwrite(savefile_path+"dilate.jpg",img)

    ret,binary=cv2.threshold(img,140,255,cv2.THRESH_BINARY)
    cv2.imshow('binary',binary)
    cv2.imwrite(savefile_path+"binary.jpg",binary)

    # cv2.imshow("img",img)

    

    # erosion = cv2.erode(img,kernel,iterations = 1)
    # cv2.imshow('erosion',erosion)
    # ret,binary=cv2.threshold(img,140,255,cv2.THRESH_BINARY)
    # cv2.imshow('binary',binary)
    count = 30
   
    

    # kernel = np.ones((30,30), np.uint8)

    # for i in range(count):
    #     result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    #     i+=1
    
    # for i in range(20):
    #     result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #     i+=1
    

    # cv2.imshow("img",img)
    # cv2.imshow("result", result)

    
    
    # cv2.imwrite(savefile_path+"new.jpg",result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img_path = "/home/user/matting/img_output/pha/pha2.jpg"
    opening(img_path)
