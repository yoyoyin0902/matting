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
import matplotlib.pyplot as plt

savefile_path = "/home/user/matting/area_calculate/save/"

def show_memory_info(hint):

	pid = os.getpid()
	p = psutil.Process(pid)
	
	info = p.memory_full_info()
	memory = info.uss / 1024. / 1024
	print('{} memory used: {} MB'.format(hint, memory))


def max2(x):
    m1 = max(x)
    x2 = x.copy()
    x2.remove(m1)
    m2 = max(x2)
    return m1,m2 


def line_Segment(img,orig):
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
            cv2.line(orig,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
			
            coordinate.append(((x0,y0),(x1,y1)))
    
    line1 = math.sqrt((coordinate[0][1][0]-coordinate[1][1][0])**2+(coordinate[0][1][1]-coordinate[1][1][1])**2)
    line2 = math.sqrt((coordinate[0][0][0]-coordinate[1][0][0])**2+(coordinate[0][0][1]-coordinate[1][0][1])**2)
        
    if(line1 > line2):
        cv2.line(orig,coordinate[0][1],coordinate[1][1],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][1][0] + coordinate[1][1][0])/2
        circle_y = (coordinate[0][1][1] + coordinate[1][1][1])/2
    else:
        cv2.line(orig,coordinate[0][0],coordinate[1][0],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][0][0] + coordinate[1][0][0])/2
        circle_y = (coordinate[0][0][1] + coordinate[1][0][1])/2
		
    cv2.circle(orig,(int(circle_x),int(circle_y)),2,(0,0,255),2)	
        # cv2.imwrite(savefile_path + filester+".jpg",img_com[i])
    gc.collect()


def circle_transform(img,orig):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 210)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M_point = cv2.moments(contours[0])
    cv2.drawContours(orig, contours, -1, (0, 0, 255), 2)
    
    center_x = int(M_point['m10']/M_point['m00'])
    center_y = int(M_point['m01']/M_point['m00'])

    drawCenter = cv2.circle(orig,(int(center_x),int(center_y)),2,(255,0,0),2)

def circle_check(contour):
    perimeter = cv2.arcLength(contour, True)  #週長
    area = cv2.contourArea(contour)  #面積
    alpha = 4*np.pi*area/(perimeter**2)
    return alpha , area, perimeter
    

if __name__ == '__main__':
    show_memory_info('initial')

    dataset_root_path = r"/home/user/matting/area_calculate/"

    image_path = os.path.join(dataset_root_path,"img")
    original_path = os.path.join(dataset_root_path,"orig_img")
    # print(original_path)

    # if image_path is None:
    #     print("no picture in there~~")
    #     return -1 
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

    contours = np.empty(len(img), dtype = object)
    contour = np.empty(len(img), dtype = object)
    parameter = np.empty(len(img), dtype = object)

    for i in range(0, len(img)):
        filester = img[i].split(".")[0]
        
        img_pha[i] = cv2.imread(join(image_path,img[i]))
        # print(img_pha[i])
        img_com[i] = cv2.imread(join(original_path,orig_img[i]))

        gray[i] = cv2.cvtColor(img_pha[i],cv2.COLOR_BGR2GRAY)

        edges[i] = cv2.Canny(gray[i], 70, 210)
        
        contours, hierarchy[i] = cv2.findContours(edges[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        parameter[i] = circle_check(contours[0])
        print('area:{0:>7,.0f} perimeter :{1:>9,.2f} alpha:{2:>7,.3f}'.format(parameter[i][1], parameter[i][2], parameter[i][0]))
        # perimeter[i] = cv2.arcLength(contours[0], True)  #週長
        # area[i] = cv2.contourArea(contours[0])
        # alpha[i] = 4*np.pi*area[i]/(perimeter[i]**2) 
        # print(parameter[i][0])

        if parameter[i][0] > 0.8:
            circle_transform(img_pha[i],img_com[i])
        else:
            line_Segment(img_pha[i],img_com[i])

        cv2.imwrite(savefile_path +filester+".jpg",img_com[i])

    del(img_pha,img_com,gray,edges,hierarchy,contours,contour,parameter)
    gc.collect()
    show_memory_info('finished')
