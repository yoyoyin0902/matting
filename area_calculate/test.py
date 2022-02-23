import os
import numpy as np 
import cv2
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict

def max2(x):
    m1 = max(x)
    x2 = x.copy()
    x2.remove(m1)
    m2 = max(x2)
    return m1,m2  

def line_Segment(img):
    
    ver_lines=[]
    coordinate=[]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(gray)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0])) 
        y0 = int(round(dline[0][1])) 
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        ver_lines.append(distance)
    maxIndex = max2(ver_lines)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        # # print(distance[i])

		# ver_lines[i].append(distance[i])
			
        if(distance >= int(maxIndex[1])):
            cv2.line(img,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
            plt.axline((x0,y0), (x1,y1), linewidth=2, color="r",marker='o')
            plt.show()
            coordinate.append(((x0,y0),(x1,y1)))
    
    # line1 = math.sqrt((coordinate[0][1][0]-coordinate[1][1][0])**2+(coordinate[0][1][1]-coordinate[1][1][1])**2)
    # line2 = math.sqrt((coordinate[0][0][0]-coordinate[1][0][0])**2+(coordinate[0][0][1]-coordinate[1][0][1])**2)
        
    # if(line1 > line2):
    #     cv2.line(img,coordinate[0][1],coordinate[1][1],(255,0,0),2,cv2.LINE_AA)
    #     circle_x = (coordinate[0][1][0] + coordinate[1][1][0])/2
    #     circle_y = (coordinate[0][1][1] + coordinate[1][1][1])/2
    # else:
    #     cv2.line(orig,coordinate[0][0],coordinate[1][0],(255,0,0),2,cv2.LINE_AA)
    #     circle_x = (coordinate[0][0][0] + coordinate[1][0][0])/2
    #     circle_y = (coordinate[0][0][1] + coordinate[1][0][1])/2
		
    # cv2.circle(img,(int(circle_x),int(circle_y)),2,(0,0,255),2)	
        # cv2.imwrite(savefile_path + filester+".jpg",img_com[i])
    cv2.imshow("img",img)


if __name__ == '__main__':

    image_path = "29_img.jpg"
    img_pha = cv2.imread(image_path)
    line_Segment(img_pha)
    cv2.waitKey(0)
	
    
    




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



 
 


