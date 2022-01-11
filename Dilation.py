#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # savepath = "/home/user/matting/binary.jpg"
# # savepath1 = "/home/user/matting/dilation.jpg"
# # /home/user/matting/pha.jpg
# # /home/user/matting/imagedata/img/2.jpg
# # /home/user/matting/imagedata/img/1.jpg
# image = cv2.imread('/home/user/matting/imagedata/img/1.jpg')
# cv2.imshow('original',image)




# img_sum = np.sum(image,axis=2)
# std_img = np.std(image)
# std_img = np.where(img_sum<255,std_img,255)
# cv2.imwrite("std_img.png",std_img.astype(np.uint8))
# plt.imshow(std_img)
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# cv2.imshow('cvtColor', img)

# cv2.imshow('Input', image)

# kernel = np.ones((3,3), np.uint8)
# dilation = cv2.erode(image, kernel, iterations = 0)
# dilation = cv2.dilate(dilation, (7,7), iterations=0)
# ret,binary=cv2.threshold(image,180,255,cv2.THRESH_BINARY)
# cv2.imshow('Input', image)
# cv2.imshow('Result', dilation)
# cv2.imshow('binary',binary)
# cv2.imwrite(savepath, binary)
# cv2.imwrite(savepath1, dilation)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os
 
import numpy as np
import matplotlib.pyplot as plt

savepath = "/home/user/matting"
 
#%%
 
def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    for y in range(0,wall.shape[0]):
        for x in range(0,wall.shape[1]):
            if wall[y,x]!=-1:
                window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num = np.amax(window)
                temp[y,x] = num
    A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
    return A
 
def min_filtering(N, A):
    wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
    temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    for y in range(0,wall_min.shape[0]):
        for x in range(0,wall_min.shape[1]):
            if wall_min[y,x]!=300:
                window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num_min = np.amin(window_min)
                temp_min[y,x] = num_min
    B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
    return B
 
def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
    return norm_img
 
def min_max_filtering(M, N, I):
    if M == 0:
        #max_filtering
        A = max_filtering(N, I)
        #min_filtering
        B = min_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
    elif M == 1:
        #min_filtering
        A = min_filtering(N, I)
        #max_filtering
        B = max_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
    return normalised_img
 
 
 
P = cv2.imread('/home/user/matting/imagedata/img/1.jpg',0)
cv2.imwrite("P.jpg",P)
 
O_P = min_max_filtering(M = 0, N = 20, I = P)
 
cv2.imwrite("O_P.jpg",O_P)
cv2.waitKey()



