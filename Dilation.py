#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
savepath = "/home/user/matting/binary.jpg"
savepath1 = "/home/user/matting/dilation.jpg"

image = cv2.imread('/home/user/matting/pha.jpg', 0)

#img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#cv2.imshow('cvtColor', img)

kernel = np.ones((3,3), np.uint8)
dilation = cv2.erode(image, kernel, iterations = 0)
dilation = cv2.dilate(dilation, (7,7), iterations=0)
ret,binary=cv2.threshold(image,180,255,cv2.THRESH_BINARY)
cv2.imshow('Input', image)
cv2.imshow('Result', dilation)
cv2.imshow('binary',binary)
cv2.imwrite(savepath, binary)
cv2.imwrite(savepath1, dilation)

cv2.waitKey(0)

