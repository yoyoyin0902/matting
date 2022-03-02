import os
import gc
import cv2
import time
import math
import psutil
import tracemalloc 
import numpy as np
from os import listdir
from os.path import isfile, join


savefile_path = "/home/user/matting/area_calculate/test/"

def FillHole(imgPath):

    # 复制 im_in 图像
    im_floodfill = imgPath.copy()
    
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = imgPath.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = imgPath | im_floodfill_inv

    return im_out
     
    # 保存结果
    # cv2.imwrite(SavePath, im_out)

def retangle_area(coordinate_0,coordinate_1,coordinate_3):
    w = coordinate_1[0] - coordinate_0[0]
    h = coordinate_3[1] - coordinate_0[1]
    area = w*h
    return area

def circle_area(radius):
    pi = 3.1415926
    area = radius * radius * pi
    return area


if __name__ == '__main__':

    dataset_root_path = r"/home/user/matting/area_calculate/"

    image_path = os.path.join(dataset_root_path,"img")
    original_path = os.path.join(dataset_root_path,"orig_img")

    img = [ f for f in listdir(image_path) if isfile(join(image_path,f))]
    orig_img = [ f for f in listdir(original_path) if isfile(join(original_path,f))]

    #排序
    img.sort(key = lambda i: int(i.rstrip('.jpg')))
    orig_img.sort(key = lambda i: int(i.rstrip('.jpg')))

    img_pha = np.empty(len(img), dtype = object)
    img_com = np.empty(len(img), dtype = object)
    gray = np.empty(len(img), dtype = object)
    edges = np.empty(len(img), dtype = object)
    hierarchy = np.empty(len(img), dtype = object)
    drawContours = np.empty(len(img), dtype = object)
    bin_image = np.empty(len(img), dtype = object)
    # floodfill = np.empty(len(img), dtype = object)
    # h = np.empty(len(img), dtype = object)
    # w = np.empty(len(img), dtype = object)
    # mask = np.empty(len(img), dtype = object)
    
    # floodfill_block = np.empty(len(img), dtype = object)
    fill_out = np.empty(len(img), dtype = object)
    img2 = np.empty(len(img), dtype = object)
    areas = np.empty(len(img), dtype = object)
    max_rect = np.empty(len(img), dtype = object)
    max_box = np.empty(len(img), dtype = object)
    retangleArea = np.empty(len(img), dtype = object)
    circleArea = np.empty(len(img), dtype = object)

    max_id = np.empty(len(img), dtype = object)
    thingarea = np.empty(len(img), dtype = object)

    pts1 = np.empty(len(img), dtype = object)
    pts2 = np.empty(len(img), dtype = object)
    M = np.empty(len(img), dtype = object)
    dst = np.empty(len(img), dtype = object)


    x = np.empty(len(img), dtype = object)
    y = np.empty(len(img), dtype = object)
    radius = np.empty(len(img), dtype = object)
    center  = np.empty(len(img), dtype = object)

    for i in range(0, len(img)):
        isbreak = False
        filester = img[i].split(".")[0]
        img_pha[i] = cv2.imread(join(image_path,img[i]))
        img_com[i] = cv2.imread(join(original_path,orig_img[i]))

        gray[i] = cv2.cvtColor(img_pha[i],cv2.COLOR_BGR2GRAY)

        ret,bin_image[i] = cv2.threshold(gray[i],127,255,cv2.THRESH_BINARY)

        fill_out[i] = FillHole(bin_image[i])

        # floodfill[i]  = bin_image[i].copy()

        # h[i], w[i] = gray[i].shape[:2]
        # mask[i] = np.zeros((h[i]+2, w[i]+2), np.uint8)
        # print(h[i],w[i])
             
        # for a in  range(floodfill[i].shape[0]):
        #     for b in range(floodfill[i].shape[1]):
        #         if(floodfill[i][a][b] == 0):
        #             seedPoint = (a,b)
        #             isbreak = True
        #             break
        #     if(isbreak):
        #         break

        # cv2.floodFill(floodfill[i], mask[i], seedPoint, 255)
        # floodfill_block[i] = cv2.bitwise_not(floodfill[i]) #要被填滿的空洞
        # fill_out[i] = bin_image[i] | floodfill_block[i] #把空洞和上面要被填滿的結合

        edges[i] = cv2.Canny(fill_out[i], 70, 210)
        contours, hierarchy[i] = cv2.findContours(edges[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))



        # contours.sort(key = cnt_area, reverse=False)
        

        
        drawContours[i] = cv2.drawContours(img_com[i], contours, -1, (0, 0, 255), 2) 

        
        areas[i] = []
        for c in range(len(contours)):
            areas[i].append(cv2.contourArea(contours[c]))

        max_id[i] = areas[i].index(max(areas[i]))
        cnt = contours[max_id[i]]
        # print(max_id[i])

        max_rect[i] = cv2.minAreaRect(contours[max_id[i]])
        max_box[i] = cv2.boxPoints(max_rect[i])
        
        #矩形輪廓面積
        retangleArea[i] = retangle_area(max_box[i][0],max_box[i][1],max_box[i][3])
        max_box[i] = np.int0(max_box[i])

        thingarea[i] = cv2.contourArea(cnt)
        print("thingarea:  ",thingarea[i])
        print("retangleArea:",retangleArea[i])
        
        
        
        img2[i] = cv2.drawContours(img_com[i],[max_box[i]],0,(0,255,0),2)

        # pts1[i] = np.float32(max_box[i])
        # pts2[i] = np.float32([[max_rect[i][0][0]+max_rect[i][1][1]/2, max_rect[i][0][1]+max_rect[i][1][0]/2],
        #               [max_rect[i][0][0]-max_rect[i][1][1]/2, max_rect[i][0][1]+max_rect[i][1][0]/2],
        #               [max_rect[i][0][0]-max_rect[i][1][1]/2, max_rect[i][0][1]-max_rect[i][1][0]/2],
        #               [max_rect[i][0][0]+max_rect[i][1][1]/2, max_rect[i][0][1]-max_rect[i][1][0]/2]])

        # print(pts1[i],pts2[i])
        # M[i] = cv2.getPerspectiveTransform(pts1[i],pts2[i])
        # dst[i] = cv2.warpPerspective(img2[i], M[i], (img2[i].shape[1],img2[i].shape[0])) 
        

        (x[i], y[i]), radius[i] = cv2.minEnclosingCircle(cnt)
        center[i] = (int(x[i]), int(y[i]))
        radius[i] = int(radius[i])
        circleArea[i] = circle_area(radius[i])
        print("circleArea:  ",circleArea[i])
        print()
        cv2.circle(img2[i], center[i], radius[i], (255, 0, 0), 2) 


        # cv2.imshow("img",img2[i])
        # cv2.waitKey(0)

        


        # print(floodfill[i].shape)
        cv2.imwrite(savefile_path +filester+"_1.jpg",fill_out[i])
        cv2.imwrite(savefile_path +filester+".jpg",img2[i])

    
    del(img_pha,img_com,gray,edges,hierarchy,drawContours,bin_image,fill_out)#
    # gc.collect()



        

        


        # edges[i] = cv2.Canny(gray[i], 70, 210)

        # contours, hierarchy[i] = cv2.findContours(edges[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))

        # drawContours[i] = cv2.drawContours(img_com[i], contours, -1, (0, 0, 255), 2) 


        # cv2.imwrite(savefile_path +filester+".jpg",bin_image[i])


 




    # img_pha = cv2.imread(image_path)

    # # cv2.imshow("111",img_pha)
    # img_com = cv2.imread(original_path)
    
    # gray = cv2.cvtColor(img_pha,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 70, 210)
    
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[13]
    # #aa = cv2.drawContours(img_com, contours, -1, (0, 0, 255), 2)
    
    # areas = []
    # print(len(contours))
    # for c in range(len(contours)):
    #     areas.append(cv2.contourArea(contours[c]))

    # max_id = areas.index(max(areas))

    # max_rect = cv2.minAreaRect(contours[max_id])
    # max_box = cv2.boxPoints(max_rect)
    # max_box = np.int0(max_box)
    # img2 = cv2.drawContours(img_com,[max_box],0,(0,255,0),2)

    # pts1 = np.float32(max_box)
    # pts2 = np.float32([[max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
    #                   [max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
    #                   [max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2],
    #                   [max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2]])

    # M = cv2.getPerspectiveTransform(pts1,pts2)
    # dst = cv2.warpPerspective(img2, M, (img2.shape[1],img2.shape[0]))  

    # #外接圓
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # center = (int(x), int(y))
    # radius = int(radius)
    # cv2.circle(img2, center, radius, (255, 0, 0), 2)


    # cv2.imshow('img2',img2)
    # #line_Segment(img_pha)
    # cv2.waitKey(0)


 

 
# max_id = areas.index(max(areas))
 
# max_rect = cv2.minAreaRect(contours[max_id])
# max_box = cv2.boxPoints(max_rect)
# max_box = np.int0(max_box)
# img2 = cv2.drawContours(img2,[max_box],0,(0,255,0),2)
 
# pts1 = np.float32(max_box)
# pts2 = np.float32([[max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
#                   [max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
#                   [max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2],
#                   [max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(img2, M, (img2.shape[1],img2.shape[0]))
 
# # 此处可以验证 max_box点的顺序
# color = [(0, 0, 255),(0,255,0),(255,0,0),(255,255,255)]
# i = 0
# for point in pts2:
#     cv2.circle(dst, tuple(point), 2, color[i], 4)
#     i+=1
 
# target = dst[int(pts2[2][1]):int(pts2[1][1]),int(pts2[2][0]):int(pts2[3][0]),:]
 
 
# cv2.imshow('img2',img2)
# cv2.imshow('dst',dst)
# cv2.imshow('target',target)
# cv2.waitKey()
# cv2.destroyAllWindows()

	
    
    




# class Solution():
#     def maxHist(self, row):
#         # Create an empty stack. The stack holds
#         # indexes of hist array / The bars stored
#         # in stack are always in increasing order
#         # of their heights.
#         result = []
 
#         # Top of stack
#         top_val = 0
 
#         # Initialize max area in current
#         max_area = 0
#         # row (or histogram)
 
#         area = 0  # Initialize area with current top
 
#         # Run through all bars of given
#         # histogram (or row)
#         i = 0
#         while (i < len(row)):
 
#             # If this bar is higher than the
#             # bar on top stack, push it to stack
#             if (len(result) == 0) or (row[result[-1]] <= row[i]):
#                 result.append(i)
#                 i += 1
#             else:
 
#                 # If this bar is lower than top of stack,
#                 # then calculate area of rectangle with
#                 # stack top as the smallest (or minimum
#                 # height) bar. 'i' is 'right index' for
#                 # the top and element before top in stack
#                 # is 'left index'
#                 top_val = row[result.pop()]
#                 area = top_val * i
 
#                 if (len(result)):
#                     area = top_val * (i - result[-1] - 1)
#                 max_area = max(area, max_area)
 
#         # Now pop the remaining bars from stack
#         # and calculate area with every popped
#         # bar as the smallest bar
#         while (len(result)):
#             top_val = row[result.pop()]
#             area = top_val * i
#             if (len(result)):
#                 area = top_val * (i - result[-1] - 1)
 
#             max_area = max(area, max_area)
 
#         return max_area
 
#     # Returns area of the largest rectangle
#     # with all 1s in A
#     def maxRectangle(self, A):
 
#         # Calculate area for first row and
#         # initialize it as result
#         result = self.maxHist(A[0])
 
#         # iterate over row to find maximum rectangular
#         # area considering each row as histogram
#         for i in range(1, len(A)):
 
#             for j in range(len(A[i])):
 
#                 # if A[i][j] is 1 then add A[i -1][j]
#                 if (A[i][j]==1):
#                     A[i][j] += int(A[i - 1][j])
 
#             # Update result if area with current
#             # row (as last row) of rectangle) is more
#             result = max(result, self.maxHist(A[i]))
 
#         return result
 
 
# if __name__ == '__main__':
#     image_path="15484244577811.jpg"
    
#     original_img = cv2.imread(image_path)
#     print(original_img.shape)

#     img = cv2.imread(image_path,0)
#     print(img.shape)

#     #ret,bin_image = cv2.threshold(img,127,1,cv2.THRESH_BINARY)
#     height,width =img.shape

#     print(height) #m
#     print(width) #n
 
#     A = [[0, 0, 0, 0],
#          [1, 1, 1, 1],
#          [1, 0, 1, 1],
#          [1, 1, 1, 0],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 0, 1, 1],
#          [1, 1, 0, 0],
#          [1, 1, 0, 0]]
#     ans = Solution()
 
#     print("Area of maximum rectangle is",
#         ans.maxRectangle(A))
#     cv2.imshow("image",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





# A = [ [0, 0, 0, 0],
#       [1, 1, 1, 1],
#       [1, 1, 1, 1],
#       [1, 1, 1, 0],
#       [1, 1, 0, 0]]
 
 
# def findMaxRectangle(nums):
#     row = len(nums)
#     _max = 0
#     for i in range(row):
#         cur_height = nums[i]  # 获取当前 height
#         cur_width = 1  # 初始化当前 width
#         # 向左寻找
#         for j in range(i - 1, -1, -1):
#             if nums[j] >= cur_height:
#                 cur_width += 1
#             else:
#                 break
#         # 向右寻找
#         for k in range(i + 1, row, 1):
#             if nums[k] >= cur_height:
#                 cur_width += 1
#             else:
#                 break
#         # area = width * height
#         now_max = cur_width * cur_height
#         _max = max(_max, now_max)
#     print(_max)
#     return _max



# def getHeightList(matrix):
#     _max = 0
#     # print("================")
#     # for i in A:
#     #     print(i)
#     # print("================")
#     if matrix is None or len(matrix) == 0:
#         return 0

#     # 获取矩阵的长，宽
#     row = len(matrix)
#     col = len(matrix[0])

#     cur_height = [0] * col
#     # 累加获取每行对应的 nums 数组
#     for i in range(row):
#         for j in range(col):
#             if (matrix[i][j]) == 1:
#                 cur_height[j] += 1
#                 print(cur_height)
#             else:
#                 cur_height[j] = 0
                
            
#         # 为每行 nums 数组计算本行的最大矩阵面积
#         cur_max = findMaxRectangle(cur_height)
#         #print(cur_max)
#         _max = max(cur_max, _max)
#     print("================")
#     return _max
 
 
# maxArea = getHeightList(A)
# print("最大面积为: %d" % maxArea)





# class Solution:
#     def maximalRectangle(self, matrix):
#         """
#         :type matrix: List[List[str]]
#         :rtype: int
#         """
#         if matrix is None or len(matrix) == 0:
#             return 0

#         height, width = len(matrix), len(matrix[0])
#         curr, result = [0]*width, 0
        
#         #print(curr)


#         for height in matrix:
#             for i in range(width):
#                 curr[i] = curr[i] + 1 if height[i] == 1 else 0
#             #print(curr)
#             result = max(self.maxHist(curr), result)
#         return result
    
#     def maxHist(self, row):
#         result = []
 
#         # Top of stack
#         top_val = 0
 
#         # Initialize max area in current
#         max_area = 0
#         # row (or histogram)
 
#         area = 0  # Initialize area with current top
 
#         # Run through all bars of given
#         # histogram (or row)
#         i = 0
        
#         #print(len(row))
#         while (i < len(row)):
 
#             # If this bar is higher than the
#             # bar on top stack, push it to stack
#             print( row[i])
#             if (len(result) == 0) or (row[result[-1]] <= row[i]):   
                
#                 result.append(i)
                
#                 i += 1
#             else:
 
#                 top_val = row[result.pop()]
#                 #print(top_val)
#                 area = top_val * i
 
#                 if (len(result)):
#                     area = top_val * (i - result[-1] - 1)
#                     #print(area)
#                 max_area = max(area, max_area)
#         while (len(result)):
#             top_val = row[result.pop()]
#             area = top_val * i
#             if (len(result)):
#                 area = top_val * (i - result[-1] - 1)
 
#             max_area = max(area, max_area)
 
#         return max_area
    

# if __name__ == '__main__':
#     image_path="15484244577811.jpg"
    
#     original_img = cv2.imread(image_path)
#     print(original_img.shape)

#     img = cv2.imread(image_path,0)
#     print(img.shape)
#     ret,bin_image = cv2.threshold(img,80,1,cv2.THRESH_BINARY)

#     #ret,bin_image = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#     height,width =bin_image.shape

#     print(height) #m
#     print(width) #n

#     A = [ [0, 0, 0, 0],
#       [1, 1, 1, 1],
#       [1, 1, 1, 1],
#       [1, 1, 1, 0],
#       [1, 1, 0, 0]]

#     # list_string  = List[List[map(str,A)]]
#     # print(A)

#     ans = Solution()
    
#     print("Area of maximum rectangle is",ans.maximalRectangle(bin_image))
#     cv2.imshow("image",original_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



 
 


