import cv2
import numpy as np
import matplotlib.pylab as plt
import time

img=cv2.imread('1927928.jpg')
# img=cv2.imread('25_com.png')

# img=img[14:-15,13:-14]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("num of contours: {}".format(len(contours)))


rect = cv2.minAreaRect(contours[0])  #获取蓝色矩形的中心点、宽高、角度
print(rect)


'''
retc=((202.82777404785156, 94.020751953125),
 (38.13406753540039, 276.02105712890625),
 -75.0685806274414)


width = int(rect[1][0])
height = int(rect[1][1])
angle = rect[2]
print(angle)

if width < height:  #计算角度，为后续做准备
  angle = angle - 90
print(angle)
'''
if rect[-1] < -45 or rect[-1] > 45:
	rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
angle = rect[2]
width = int(rect[1][0])
height = int(rect[1][1])
# if  angle < -45:
#     angle += 90.0
#        #保证旋转为水平
# width,height = height,width
src_pts = cv2.boxPoints(rect)

# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(img_box, [box], 0, (0,255,0), 2) 
#

dst_pts = np.array([[0, height],
                    [0, 0],
                    [width, 0],
                    [width, height]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (1680, 867))

if angle<=-90:  #对-90度以上图片的竖直结果转正
    warped = cv2.transpose(warped)
    warped = cv2.flip(warped, 0)  # 逆时针转90度，如果想顺时针，则0改为1
    # warped=warped.transpose
cv2.imshow('wr1',warped)
cv2.waitKey(0)

