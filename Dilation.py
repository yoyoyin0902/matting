#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/home/user/matting/img_output/pha.jpg', 0)
kernel = np.ones((9,9), np.uint8)
dilation = cv2.erode(image, kernel, iterations = 1)
ret,binary=cv2.threshold(image,127,0,cv2.THRESH_BINARY)
cv2.imshow('Input', image)
cv2.imshow('Result', dilation)
cv2.imshow('binary',binary)
cv2.waitKey(0)
