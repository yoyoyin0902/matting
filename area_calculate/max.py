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


'''
Script that finds largest rectangle in a binary image
Method : We will process the image by rows,
for each row we define first line of pixels containing "ones" that we 
evaluate over the vertical axis to get the largest rectangle possible
Author : Robert Kelevra
Requirements opencv, python3
'''

'''Give image path and run script'''




# def func():
# 	show_memory_info('initial')
# 	a = [i for i in range(10000000)]
# 	show_memory_info('after a created')



#function that computes rectangle area
def compute_area(ymax,ymin,xmax,xmin):
	return (ymax-ymin)*(xmax-xmin)
	
# Coordinates of pixels that define our rectangle
coord_upper_left={"x":0,"y":0}
coord_lower_left={"x":0,"y":0}
coord_upper_right={"x":0,"y":0}
coord_lower_right={"x":0,"y":0}
xmax=0
xmin=0
ymax=0
ymin=0
#Booleans that track bounds as to not take into account zero pixels
upper_left=False
lower_left=False

#Track of continous rectangles using unhindered points
#unhindered stores bound that need to be evaluated
unhindered=[]

#function that reset all variables for row analysis
#used after every row evaluation
def reset_variables():
	global	coord_upper_left
	global	coord_lower_left
	global	coord_upper_right
	global	coord_lower_right
	global	upper_left
	global	lower_left
	global	unhindered
	
	coord_upper_left={"x":0,"y":0}
	coord_lower_left={"x":0,"y":0}
	coord_upper_right={"x":0,"y":0}
	coord_lower_right={"x":0,"y":0}
	upper_left=False
	lower_left=False
	unhindered=[]


# def find_max_rectangle1(image_path,original_path):
# 	start=time.time()
# 	global	coord_upper_left
# 	global	coord_lower_left
# 	global	coord_upper_right
# 	global	coord_lower_right
# 	global	upper_left
# 	global	lower_left
# 	global	unhindered
# 	global  real_image

# 	#Initial area of our first rectangle
# 	area=0
# 	if image_path is None:
# 		print("no picture in there~~")
# 		return -1 
	
# 	img = [ f for f in listdir(image_path) if isfile(join(image_path,f))]
# 	orig_img = [ f for f in listdir(original_path) if isfile(join(original_path,f))]
# 	print(img)


# 	img.sort(key = lambda i: int(i.rstrip('.jpg')))
# 	orig_img.sort(key = lambda i: int(i.rstrip('.jpg')))
# 	print(img)


# 	img_count = len(img)

# 	img_pha = np.empty(img_count, dtype=object)
# 	img_com = np.empty(img_count, dtype=object)
# 	gray = np.empty(img_count, dtype=object)

# 	bin_image = np.empty(img_count, dtype=object)
# 	height =  np.empty(img_count, dtype=object)
# 	width =  np.empty(img_count, dtype=object)

# 	xmax = np.empty(img_count, dtype=object)
# 	xmin = np.empty(img_count, dtype=object)
# 	ymax = np.empty(img_count, dtype=object)
# 	ymin = np.empty(img_count, dtype=object)



# 	for i in range(0,len(img)):
# 		filester = img[i].split(".")[0]

# 		img_pha[i] = cv2.imread(join(image_path,img[i]))
# 		img_com[i] = cv2.imread(join(original_path,orig_img[i]))

# 		gray[i] = cv2.cvtColor(img_pha[i],cv2.COLOR_BGR2GRAY)
# 		ret,bin_image[i] = cv2.threshold(gray[i],128,1,cv2.THRESH_BINARY)

# 		height[i], width[i] = bin_image[i].shape
# 		# print(height[i], width[i])
# 		for a in range(height[i]):
# 			reset_variables()

# 			if(area>(width[i]*(height[i]-a))):
# 				break

# 			for b in range(width[i]):
# 				if(bin_image[i][a,b]==0 and not(upper_left)):
# 					continue   #跳出這個迴圈
				
# 				#找到定義左上角坐標的“1”像素
# 				if(bin_image[i][a,b]==1 and not(upper_left)):
# 					coord_upper_left["x"]=b
# 					coord_upper_left["y"]=a
# 					if(b==(width[i]-1)):
# 						coord_upper_right["y"]=a
# 						coord_upper_right["x"]=b
# 					upper_left=True
# 				#定義右上座標
# 				if((bin_image[i][a,b]==0 and upper_left) or (bin_image[i][a,b]==1 and b==(width[i]-1) and upper_left)):
# 					coord_upper_right["x"]=b-1
# 					coord_upper_right["y"]=a
# 					if(b==(width[i]-1)):
# 						coord_upper_right["x"]=b
# 					upper_left=False

# 					for horizontal_counter in (coord_upper_left["x"],(coord_upper_right["x"]+1)):
# 						for vertical_counter in range((a+1),height[i]):
# 							if(bin_image[i][vertical_counter,horizontal_counter]==0 and not lower_left):
# 								lower_left=True
# 								coord_lower_left["x"]=horizontal_counter
# 								coord_lower_left["y"]=vertical_counter-1

# 								w=vertical_counter-coord_upper_left["y"]
# 								if(w>area):
# 									area=w
# 									ymax[i]=height[i]-coord_upper_left["y"]
# 									ymin[i]=height[i]-coord_lower_left["y"]-1
# 									xmax[i]=coord_lower_left["x"]+1
# 									xmin[i]=coord_lower_left["x"]
# 								length_unhindered=len(unhindered)
								
	# del(img_pha,img_com,gray,bin_image,height,width,xmax,xmin,ymax,ymin)
	# gc.collect()

def find_max_rectangle(image_path,original_path):
	start=time.time()
	global	coord_upper_left
	global	coord_lower_left
	global	coord_upper_right
	global	coord_lower_right
	global	upper_left
	global	lower_left
	global	unhindered
	global  real_image
	#Initial area of our first rectangle
	area=0

	#get image, binarize it if necessary
	old_img = cv2.imread(original_path)
	real_image=cv2.imread(image_path)
	image=cv2.imread(image_path,0)
	ret,bin_image = cv2.threshold(image,128,1,cv2.THRESH_BINARY)
		
	#Shape of image
	height, width = bin_image.shape
	for i in range(height):

		#Reset all variables
		reset_variables()	
		#Dealing with the case of last row 

		#Check that our current area is larger than the max of area remaining to check
		#If so no need to continue
		if(area>(width*(height-i))):
			break
				
		for j in range(width):
			#Find the first line of pixels containing "1"
			if(bin_image[i,j]==0 and not(upper_left)):
				# Do not consider pixels that are equal to 0 unless upper left bound is ddefined
				continue
			if(bin_image[i,j]==1 and not(upper_left)):
				#We found our "1" pixel that defines our upper left coordinate
				coord_upper_left["x"]=j
				coord_upper_left["y"]=i
				if(j==(width-1)):
					coord_upper_right["y"]=i
					coord_upper_right["x"]=j
				upper_left=True
			#define our upper right coordinate after upper left coordinate has been set
			if((bin_image[i,j]==0 and upper_left) or (bin_image[i,j]==1 and j==(width-1) and upper_left) ):
				coord_upper_right["x"]=j-1
				coord_upper_right["y"]=i
				if(j==(width-1)):
					coord_upper_right["x"]=j
				upper_left=False
				
				#Vertical evaluation of previously found line through rows
				#Horizontal and vertical counters for evaluation
				for horizontal_counter in range(coord_upper_left["x"],(coord_upper_right["x"]+1)):
					for vertical_counter in range((i+1),height):
						#iteratively check rectangles using lower left tracker
						#we hit a bound when we meet a '0' pixel or we hit the height
						if(bin_image[vertical_counter,horizontal_counter]==0 and not(lower_left)):
							lower_left=True
							coord_lower_left["x"]=horizontal_counter
							coord_lower_left["y"]=vertical_counter-1
							#compute the area for this particular case
							a=vertical_counter-coord_upper_left["y"]
							#check to see if a larger area exists
							#if so set rectangle coordinates
							if(a>area):
								area=a
								ymax=height-coord_upper_left["y"]
								ymin=height-coord_lower_left["y"]-1
								xmax=coord_lower_left["x"]+1
								xmin=coord_lower_left["x"]
							#No need to continue downward, we have our first vertical line
							#so we break the vertical counter loop
							break
						#if we hit the bottom and we find no lower left bound
						#we set a lower left coordinate to last element of the vertical line
						if(vertical_counter==height-1 and bin_image[vertical_counter,horizontal_counter]==1 and not(lower_left)):
							lower_left=True
							coord_lower_left["x"]=horizontal_counter
							coord_lower_left["y"]=vertical_counter
							#compute area and compare it 
							a=height-coord_upper_left["y"]
							if(a>area):
								area=a
								ymax=height-coord_upper_left["y"]
								ymin=height-coord_lower_left["y"]-1
								xmax=coord_lower_left["x"]+1
								xmin=coord_lower_left["x"]
							break
						#lower left coordinate has already been set
						#so we are basically checking vertical lines along our initial pixel line at the top
						if((bin_image[vertical_counter,horizontal_counter]==0 and lower_left)):
							if(coord_lower_left["y"]<vertical_counter-1):
							#we went lower than the what had already ben set as a lower left bound
							#so a new unhindered element has to be created 
							#UNHINDERED RECTANGLES are those that continue to grow horizontally without meeting an obstacle
							#unhindered elements contain current lower left coordinates as well as upper left coordinates
							#they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
								len_unhindered=len(unhindered)
								#we do not want to make hindered rectangles unhindered so we have to check that they are not already set
								already=False
								for l in range(len_unhindered):
									if(unhindered[l][0]==coord_lower_left["y"]):
										already=True
								if(not(already)):		
									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
								#we set new lower bounds and upper bounds accordingly
								coord_lower_left["x"]=horizontal_counter
								coord_lower_left["y"]=vertical_counter-1
								coord_upper_left["x"]=horizontal_counter
								#upper counter "y" coordinate remains the same
								#compute the area and compare it accordignly
								a=vertical_counter-coord_upper_left["y"]
								if(a>area):
									area=a
									ymax=height-coord_upper_left["y"]
									ymin=height-coord_lower_left["y"]-1
									xmax=horizontal_counter+1
									xmin=coord_upper_left["x"]
								#Now we compute areas of above unhindered rectangles
								#and compare their areas
								length_unhindered=len(unhindered)
								
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							#We remained higher than our lower left bound that we had set
							#we therefore get rid of unhindered rectangles that do not need 
							#to be considered because they have become hindered
							#new unhindered are also created
							if((coord_lower_left["y"]>(vertical_counter-1))):
								#first insert a new unhindered element lower bound between current hindered elements
								#correct particular exceptions
								length_unhindered=len(unhindered)
								checked=False
								added=False
								for l in range(length_unhindered):
									if(unhindered[l][0]<vertical_counter-1):
										checked=True
									if(unhindered[l][0]>vertical_counter-1):
										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
										added=True
										break
								if(checked and not(added)):
									unhindered.append([(vertical_counter-1),(coord_lower_left["x"]),coord_upper_left["y"],coord_lower_left["x"]])
								#now get rid of hindered elements
								length_unhindered=len(unhindered)
								if(length_unhindered!=0):
									indices=[]
									for l in range(length_unhindered):
										if(unhindered[l][0]>(vertical_counter-1)):
											indices.append(l)
									indices.reverse()
									for indice in indices:
										unhindered.pop(indice)
								#compute remaining areas
								#first check to see if there were indeed unhidered elements previously created
								#compute and compare their areas
								if(length_unhindered!=0):
									length_unhindered=len(unhindered)
									coord_lower_left["y"]=vertical_counter-1
									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
									for l in range(length_unhindered):
										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
										if(unhindered_area>area):
											area=unhindered_area
											ymax=(height-unhindered[l][2])
											ymin=(height-unhindered[l][0]-1)
											xmax=(horizontal_counter+1)
											xmin=unhindered[l][1]
										
								if(length_unhindered==0):
									coord_lower_left["y"]=vertical_counter-1
									#compute one area
									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
									if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								break
							#if we stay at the same lower bound
							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
								#compute and compare
								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
								if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								#check unhindered elements
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							break
						#Special case where we hit the bottom
						if((bin_image[vertical_counter,horizontal_counter]==1 and lower_left and vertical_counter==(height-1))):
							if(coord_lower_left["y"]<vertical_counter):
							#we went lower than the what had already ben set as a lower left bound
							#so a new unhindered element has to be created 
							#unhindered elements contain current lower left coordinates as well as upper left coordinates
							#they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
								len_unhindered=len(unhindered)
								already=False
								for l in range(len_unhindered):
									if(unhindered[l][0]==coord_lower_left["y"]):
										already=True
								if(not(already)):		
									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
								#we set new lower bounds and upper bounds accordingly
								coord_lower_left["x"]=horizontal_counter
								coord_lower_left["y"]=vertical_counter
								coord_upper_left["x"]=horizontal_counter
								#upper counter "y" coordinate remains the same
								#compute the area and compare it accordignly
								a=height-coord_upper_left["y"]
								if(a>area):
									area=a
									ymax=height-coord_upper_left["y"]
									ymin=height-coord_lower_left["y"]-1
									xmax=horizontal_counter+1
									xmin=coord_upper_left["x"]
								#Now we compute areas of above unhindered rectangles
								#and compare their areas
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							#We remained higher than our lower left bound that we had set
							#we therefore get rid of unhindered elements (rectangles) that do not need 
							#to be considered because they have become hindered
							if((coord_lower_left["y"]>(vertical_counter))):
								#first insert a new unhindered element lower bound between current hindered elements
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									if(unhindered[l][0]>vertical_counter-1):
										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
										break
								#now get rid of hindered elements
								length_unhindered=len(unhindered)
								if(length_unhindered!=0):
									indices=[]
									for l in range(length_unhindered):
										if(unhindered[l][0]>(vertical_counter-1)):
											indices.append(l)
									indices.reverse()
									for indice in indices:	
										unhindered.pop(indice)
								#compute remaining areas
								#first check to see if there were indeed unhidered elements previously created
								#compute and compare their areas
								if(length_unhindered!=0):
									length_unhindered=len(unhindered)
									coord_lower_left["y"]=vertical_counter-1
									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
									for l in range(length_unhindered):
										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
										if(unhindered_area>area):
											area=unhindered_area
											ymax=(height-unhindered[l][2])
											ymin=(height-unhindered[l][0]-1)
											xmax=(horizontal_counter+1)
											xmin=unhindered[l][1]
										
								if(length_unhindered==0):
									coord_lower_left["y"]=vertical_counter-1
									#compute one area
									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
									if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								break
							#if we stay at the same lower bound
							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
								#compute and compare
								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))

								if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								#check unhindered elements
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]				
								break
							break
				reset_variables()
	end=time.time()
	print("the process took %lf seconds" %(end-start))
	cv2.rectangle(old_img, (xmin,height-ymax), (xmax-1,height-ymin-1),(0,255,0), thickness=1, lineType=8, shift=0)
	
	results = [xmin,(height-ymax),(xmax-1),(height-ymin-1)]
	rect_width = (xmax-1) - xmin
	rect_height = (height-ymin-1) - (height-ymax)
	print(rect_width)
	print(rect_height)
	area = rect_width * rect_height
	print(area)
	center_X = xmin + int(rect_width/2)
	center_Y = (height-ymax) + int(rect_height/2)
	# print(center_X)
	# print(center_Y)

	aa =cv2.circle(old_img, (center_X,center_Y), 2,(255,0,0),2)
	
	cv2.imshow("image",old_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	
	return results

def max2(x):
    m1 = max(x)
    x2 = x.copy()
    x2.remove(m1)
    m2 = max(x2)
    return m1,m2  

def line_Segment(img,orig):
	

	if image_path is None:
		print("no picture in there~~")
		return -1 

	img = [ f for f in listdir(image_path) if isfile(join(image_path,f))]
	orig_img = [ f for f in listdir(original_path) if isfile(join(original_path,f))]

	#排序
	img.sort(key = lambda i: int(i.rstrip('.jpg')))
	orig_img.sort(key = lambda i: int(i.rstrip('.jpg')))
	
	img_count = len(img)
	
	img_pha = np.empty(img_count, dtype=object)
	img_com = np.empty(img_count, dtype=object)
	gray = np.empty(img_count, dtype=object)
	dlines = np.empty(img_count, dtype=object)
	lsd = np.empty(img_count, dtype=object)
	dline = np.empty(img_count, dtype=object)
	maxIndex = np.empty(img_count, dtype=object)

	x0 = np.empty(img_count, dtype=object)
	y0 = np.empty(img_count, dtype=object)
	x1 = np.empty(img_count, dtype=object)
	y1 = np.empty(img_count, dtype=object)
	distance = np.empty(img_count, dtype=object)
	coordinate = np.empty(img_count, dtype=object)
	ver_lines = np.empty(img_count, dtype=object)

	circle_x = np.empty(img_count, dtype=object)
	circle_y = np.empty(img_count, dtype=object)

	line1 = np.empty(img_count, dtype=object)
	line2 = np.empty(img_count, dtype=object)
	
	for i in range(0,len(img)):
		filester = img[i].split(".")[0]
		
		img_pha[i] = cv2.imread(join(image_path,img[i]))
		img_com[i] = cv2.imread(join(original_path,orig_img[i]))

		gray[i] = cv2.cvtColor(img_pha[i],cv2.COLOR_BGR2GRAY)

		lsd[i] = cv2.createLineSegmentDetector(0)
	
		dlines[i] = lsd[i].detect(gray[i])

		ver_lines[i] = []
		coordinate[i] = []

		for dline[i] in dlines[i][0]:
			# print(dline[i])
			x0[i] = int(round(dline[i][0][0]))
			y0[i] = int(round(dline[i][0][1]))
			x1[i] = int(round(dline[i][0][2]))
			y1[i] = int(round(dline[i][0][3]))
			distance[i] = math.sqrt((x0[i]-x1[i])**2+(y0[i]-y1[i])**2)
			ver_lines[i].append(distance[i])

		maxIndex[i] = max2(ver_lines[i])

		for dline[i] in dlines[i][0]:
			# # print(dline[i])
			x0[i] = int(round(dline[i][0][0]))
			y0[i] = int(round(dline[i][0][1]))
			x1[i] = int(round(dline[i][0][2]))
			y1[i] = int(round(dline[i][0][3]))
			distance[i] = math.sqrt((x0[i]-x1[i])**2+(y0[i]-y1[i])**2)
			# # print(distance[i])

			# ver_lines[i].append(distance[i])
			
			if(distance[i] >= int(maxIndex[i][1])):
				cv2.line(img_com[i],(x0[i],y0[i]),(x1[i],y1[i]),(0,255,0),2,cv2.LINE_AA)
			
				coordinate[i].append(((x0[i],y0[i]),(x1[i],y1[i])))

		# print(coordinate[i])

		line1[i] = math.sqrt((coordinate[i][0][1][0]-coordinate[i][1][1][0])**2+(coordinate[i][0][1][1]-coordinate[i][1][1][1])**2)
		

		line2[i] = math.sqrt((coordinate[i][0][0][0]-coordinate[i][1][0][0])**2+(coordinate[i][0][0][1]-coordinate[i][1][0][1])**2)
			
		if(line1[i] > line2[i]):
			cv2.line(img_com[i],coordinate[i][0][1],coordinate[i][1][1],(255,0,0),2,cv2.LINE_AA)
			circle_x[i] = (coordinate[i][0][1][0] + coordinate[i][1][1][0])/2
			circle_y[i] = (coordinate[i][0][1][1] + coordinate[i][1][1][1])/2

		else:
			cv2.line(img_com[i],coordinate[i][0][0],coordinate[i][1][0],(255,0,0),2,cv2.LINE_AA)
			circle_x[i] = (coordinate[i][0][0][0] + coordinate[i][1][0][0])/2
			circle_y[i] = (coordinate[i][0][0][1] + coordinate[i][1][0][1])/2
		
	
		cv2.circle(img_com[i],(int(circle_x[i]),int(circle_y[i])),2,(0,0,255),2)	

		cv2.imwrite(savefile_path + filester+".jpg",img_com[i])
	
	del(img_pha,img_com,gray,dlines,lsd,dline,maxIndex,x0,y0,x1,y1,distance,coordinate,circle_x,circle_y,line1,line2)
	gc.collect()

#circle_transform
def circle_transform(image_path,original_path):

	if image_path is None:
		print("no picture in there~~")
		return -1 

	img = [ f for f in listdir(image_path) if isfile(join(image_path,f))]
	orig_img = [ f for f in listdir(original_path) if isfile(join(original_path,f))]
	print(orig_img)

	#排序
	img.sort(key = lambda i: int(i.rstrip('.jpg')))
	orig_img.sort(key = lambda i: int(i.rstrip('.jpg')))
	
	img_count = len(img)

	img_pha = np.empty(img_count, dtype=object)
	img_com = np.empty(img_count, dtype=object)
	gray = np.empty(img_count, dtype=object)
	edges = np.empty(img_count, dtype=object)
	hierarchy = np.empty(img_count, dtype=object)
	center_x = np.empty(img_count,dtype=object)
	center_y = np.empty(img_count,dtype=object)
	M_point = np.empty(img_count,dtype=object)

	for i in range(0,len(img)):
		filester = img[i].split(".")[0]
		
		img_pha[i] = cv2.imread(join(image_path,img[i]))
		img_com[i] = cv2.imread(join(original_path,orig_img[i]))

		gray[i] = cv2.cvtColor(img_pha[i],cv2.COLOR_BGR2GRAY)
		edges[i] = cv2.Canny(gray[i], 70, 210)

		contours, hierarchy[i] = cv2.findContours(edges[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		M_point = cv2.moments(contours[0])
		# print(M_point)
		cv2.drawContours(img_com[i], contours, -1, (0, 0, 255), 2)
		

		center_x[i] = int(M_point['m10']/M_point['m00'])
		center_y[i] = int(M_point['m01']/M_point['m00'])
		drawCenter = cv2.circle(img_com[i],(int(center_x[i]),int(center_y[i])),2,(255,0,0),2)

		# cv2.imshow('src', img_pha[i])
	
		cv2.imwrite(savefile_path + filester+".jpg",img_com[i])

	del(img_pha,img_com,gray,edges,hierarchy,center_x,center_y,M_point)
	gc.collect()


def transform(image_path,original_path):
	if image_path is None:
		print("no picture in there~~")
		return -1 

	img = [ f for f in listdir(image_path) if isfile(join(image_path,f))]
	orig_img = [ f for f in listdir(original_path) if isfile(join(original_path,f))]
		

	#排序
	img.sort(key = lambda i: int(i.rstrip('.jpg')))
	orig_img.sort(key = lambda i: int(i.rstrip('.jpg')))

	img_count = len(img)
	img_pha = np.empty(img_count, dtype=object)
	img_com = np.empty(img_count, dtype=object)

	
	for i in range(0,len(img)):
		filester = img[i].split(".")[0]
		
		img_pha[i] = cv2.imread(join(image_path,img[i]))
		img_com[i] = cv2.imread(join(original_path,orig_img[i]))

	del(img_pha,img_com)




	#-----------------------------------------one_picture-------------------------------------------------#
	img = cv2.imread(image_path)
	orig_img = cv2.imread(original_path)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# cv2.imshow("img",img)
	# circle_transform(gray)
	edges = cv2.Canny(img,50,255) 

	lsd = cv2.createLineSegmentDetector(0)

	dlines = lsd.detect(gray)
	print(dlines)
	#drawn_img = lsd.drawSegments(img,dlines)
	# print(dlines[0][0][0])

	ver_lines = []
	coordinate 	= []

	max_value = None
	
	for dline in dlines[0]:
		x0 = int(round(dline[0][0]))
		y0 = int(round(dline[0][1]))
		x1 = int(round(dline[0][2]))
		y1 = int(round(dline[0][3]))
		distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
		print(distance)
		
		if(distance >=100):
			cv2.line(orig_img[i],(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
			
			coordinate.append(((x0,y0),(x1,y1)))

		
	cv2.line(orig_img,coordinate[0][0],coordinate[1][0],(255,0,0),2,cv2.LINE_AA)
	circle_x = (coordinate[0][0][0] + coordinate[1][0][0])/2
	circle_y = (coordinate[0][0][1] + coordinate[1][0][1])/2
	print(circle_x)
	print(circle_y)
	cv2.circle(orig_img,(int(circle_x),int(circle_y)),2,(0,0,255),2)
	cv2.imshow("LSD", orig_img)
	#-----------------------------------------one_picture-------------------------------------------------#

# def circle_transform(gray_img):
# 	cv2.imshow('res',gray_img)
# 	cv2.waitKey(0)

# 	circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
# 	print(circles)
# 	# print(len(circles[0]))
# 	for circle in circles[0]:
# 		print(circle[2])
# 		x = int(circle[0])
# 		y = int(circle[1])
# 		r = int(circle[2])
# 		img=cv2.circle(gray_img,(x,y),r,(0,0,255),-1)
# 	cv2.imshow('res',img)

	
savefile_path = "/home/user/matting/area_calculate/save/"

if __name__ == '__main__':

	# tracemalloc.start() 
	# my_complex_analysis_method() 
	show_memory_info('initial')

	dataset_root_path = r"/home/user/matting/area_calculate/"
	# image_path = "29_pha.jpg"
	# original_path = "29.jpg"

	image_path = os.path.join(dataset_root_path,"img")
	original_path = os.path.join(dataset_root_path,"orig_img")


	# circle_transform(image_path,original_path)
	# line_Segment(image_path,original_path)
	find_max_rectangle1(image_path,original_path)

	# show_memory_info('after a created')
	show_memory_info('finished')
	# transform(image_path,original_path)
	# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	# tracemalloc.stop() 
	# a=find_max_rectangle(image_path)
	# print(a)
	# cv2.imshow("image",real_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



	

	









