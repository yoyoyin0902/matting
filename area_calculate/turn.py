import cv2
import numpy as np
import matplotlib.pylab as plt
import time


def homography(Original_imgPath):
  Orig_img = cv2.imread(Original_imgPath)
  row, cols = Orig_img.shape[:2]

  #灰階處理
  gray = cv2.cvtColor(Orig_img,cv2.COLOR_BGR2GRAY)
  #二值化
  ret,binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
  #找尋輪廓並繪製輪廓(二值化過後才可用)
  contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  cv2.drawContours(img,contours,-1,(0,0,255),3) 
# cv2.imshow('binary2',img)

  #尋找輪廓中心點並繪製中心點
  M = cv2.moments(contours[0])
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])
  cv2.circle(img, (cx,cy), 2,(255,0,0),2)

  cv2.imshow('get_center',img)

  


#15483951094941.jpg
img=cv2.imread('1928832.jpg')

# img=cv2.imread('25_com.png')
rows, cols = img.shape[:2] #row 867 col 1680

#轉灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#二值化
ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

newImage = binary.copy()
#找尋輪廓(二值化過後才可用)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
#繪製輪廓()
cv2.drawContours(img,contours,-1,(0,0,255),3) 
# cv2.imshow('binary2',img)

# M = cv2.moments(contours[0])
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# print(cx)
# print(cy)

print("num of contours: {}".format(len(contours)))

#得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
rect = cv2.minAreaRect(contours[0])  #获取蓝色矩形的中心点、宽高、角度

angle = rect[-1]
width = int(rect[1][0])
height = int(rect[1][1])
cx = rect[0][0]
cy = rect[0][1]
print(angle)

# cv2.circle(img, (int(cx),int(cy)), 2,(255,0,0),2)

# cv2.imshow('circle',img)

# if rect[-1] < -45 or rect[-1] > 45:
# 	rect = (rect[0], (rect[1][0], rect[1][1]), rect[2] - 90)
# else :
#   rect = (rect[0], (rect[1][0], rect[1][1]), rect[2])

# print(angle)
# print(width)
# print(height)
# print(cx)
# print(cy)
# print(angle)

#矩形四個頂點
box = cv2.boxPoints(rect) 
# print(box)

boxlist = np.int0(box) #numpy.int64
# print(boxlist)
cv2.drawContours(img, [boxlist], 0, (0, 255, 0), 1)
# cv2.imshow('box',img)

# x,y,w,h = cv2.boundingRect(rect) #（x,y）是旋转的边界矩形左上角的点，w ,h分别是宽和高
# print(x)
# print(y)

'''
width = int(rect[1][0])
height = int(rect[1][1])
angle = rect[2]
print(angle)

if width < height:  #计算角度，为后续做准备
  angle = angle - 90
print(angle)
'''


# if  angle < -45:
#     angle += 90.0
#        #保证旋转为水平
# width,height = height,width

# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(img_box, [box], 0, (0,255,0), 2) 
#

dst_pts = np.array([[0, height],
                    [0, 0],
                    [width, 0],
                    [width, height]], dtype="float32")

dst_pts1 = np.array([[cx-width/2, cy+height/2],
                    [cx-width/2, cy-height/2],
                    [cx+width/2, cy-height/2],
                    [cx+width/2, cy+height/2]], dtype="float32")
M = cv2.getPerspectiveTransform(box, dst_pts1)
warped = cv2.warpPerspective(img, M, (cols, rows))

if angle<=-90:  #对-90度以上图片的竖直结果转正
    warped = cv2.transpose(warped)
    warped = cv2.flip(warped, 1)  # 逆时针转90度，如果想顺时针，则0改为1
    warped=warped.transpose
cv2.imwrite("save3.jpg",warped)
cv2.imshow('wr1',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

