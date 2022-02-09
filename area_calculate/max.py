import cv2
import time

'''
Script that finds largest rectangle in a binary image
Method : We will process the image by rows,
for each row we define first line of pixels containing "ones" that we 
evaluate over the vertical axis to get the largest rectangle possible
Author : Robert Kelevra
Requirements opencv, python3
'''

'''Give image path and run script'''




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

def find_max_rectangle(image_path):
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
	cv2.rectangle(real_image, (xmin,height-ymax), (xmax-1,height-ymin-1),(0,255,0), thickness=1, lineType=8, shift=0)
	
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

	aa =cv2.circle(real_image, (center_X,center_Y), 2,(255,0,0),2)
	
	# cv2.imshow("image",real_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	
	return results

if __name__ == '__main__':

	image_path="15484244577811.jpg"
	a=find_max_rectangle(image_path)
	print(a)
	cv2.imshow("image",real_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# if __name__ == '__main__':
# 	# dataset_root_path = r"/home/user/matting/img_output/"
# 	# image_path = os.path.join(dataset_root_path,"com")
# 	# print(image_path)
# 	image_path="save2.jpg"
# 	#maxmalRectangle(image_path)
# 	# turn(image_path)
# 	a = find_max_rectangle(image_path)
# 	print(a)

	
# 	# b = getinformation(a)
# 	# #cv2.imshow("image",real_image)
# 	cv2.waitKey(0)
	









# import os
# import cv2
# import numpy as np
# import time


# # image_path="15483988745385.jpg"


# def compute_area(ymax,ymin,xmax,xmin):
# 	return (ymax-ymin)*(xmax-xmin)
	
# # Coordinates of pixels that define our rectangle
# coord_upper_left={"x":0,"y":0}
# coord_lower_left={"x":0,"y":0}
# coord_upper_right={"x":0,"y":0}
# coord_lower_right={"x":0,"y":0}
# xmax=0
# xmin=0
# ymax=0
# ymin=0
# #Booleans that track bounds as to not take into account zero pixels
# upper_left=False
# lower_left=False

# #Track of continous rectangles using unhindered points
# #unhindered stores bound that need to be evaluated
# unhindered=[]

# #function that reset all variables for row analysis
# #used after every row evaluation
# def init_variables():
# 	global	coord_upper_left
# 	global	coord_lower_left
# 	global	coord_upper_right
# 	global	coord_lower_right
# 	global	upper_left
# 	global	lower_left
# 	global	unhindered
	
# 	coord_upper_left={"x":0,"y":0}
# 	coord_lower_left={"x":0,"y":0}
# 	coord_upper_right={"x":0,"y":0}
# 	coord_lower_right={"x":0,"y":0}
# 	upper_left=False
# 	lower_left=False
# 	unhindered=[]

# def find_max_rectangle(image_path):
# 	start=time.time()
# 	#全域變數
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

# 	#get image, binarize it if necessary
# 	original_img=cv2.imread(image_path)
# 	real_image=cv2.imread(image_path,0)
# 	print(real_image.shape)
# 	# image=cv2.imread(image_path,0)
# 	# ret,bin_image = cv2.threshold(image,128,1,cv2.THRESH_BINARY)
			
# 	#Shape of image
# 	height, width = real_image.shape
# 	for i in range(height):
# 		#Reset all variables
# 		init_variables()	
# 		#Dealing with the case of last row 

# 		#Check that our current area is larger than the max of area remaining to check
# 		#If so no need to continue
# 		if(area>(width*(height-i))):
# 			break
				
# 		for j in range(width):
# 			#Find the first line of pixels containing "1"
# 			if(real_image[i,j]==0 and not(upper_left)):
# 				# Do not consider pixels that are equal to 0 unless upper left bound is ddefined
# 				continue
# 			if(real_image[i,j]==1 and not(upper_left)):
# 				#We found our "1" pixel that defines our upper left coordinate
# 				coord_upper_left["x"]=j
# 				coord_upper_left["y"]=i
# 				if(j==(width-1)):
# 					coord_upper_right["y"]=i
# 					coord_upper_right["x"]=j
# 				upper_left=True
# 			#define our upper right coordinate after upper left coordinate has been set
# 			if((real_image[i,j]==0 and upper_left) or (real_image[i,j]==1 and j==(width-1) and upper_left) ):
# 				coord_upper_right["x"]=j-1
# 				coord_upper_right["y"]=i
# 				if(j==(width-1)):
# 					coord_upper_right["x"]=j
# 				upper_left=False
				
# 				#Vertical evaluation of previously found line through rows
# 				#Horizontal and vertical counters for evaluation
# 				for horizontal_counter in range(coord_upper_left["x"],(coord_upper_right["x"]+1)):
# 					for vertical_counter in range((i+1),height):
# 						#iteratively check rectangles using lower left tracker
# 						#we hit a bound when we meet a '0' pixel or we hit the height
# 						if(real_image[vertical_counter,horizontal_counter]==0 and not(lower_left)):
# 							lower_left=True
# 							coord_lower_left["x"]=horizontal_counter
# 							coord_lower_left["y"]=vertical_counter-1
# 							#compute the area for this particular case
# 							a=vertical_counter-coord_upper_left["y"]
# 							#check to see if a larger area exists
# 							#if so set rectangle coordinates
# 							if(a>area):
# 								area=a
# 								ymax=height-coord_upper_left["y"]
# 								ymin=height-coord_lower_left["y"]-1
# 								xmax=coord_lower_left["x"]+1
# 								xmin=coord_lower_left["x"]
# 							#No need to continue downward, we have our first vertical line
# 							#so we break the vertical counter loop
# 							break
# 						#if we hit the bottom and we find no lower left bound
# 						#we set a lower left coordinate to last element of the vertical line
# 						if(vertical_counter==height-1 and real_image[vertical_counter,horizontal_counter]==1 and not(lower_left)):
# 							lower_left=True
# 							coord_lower_left["x"]=horizontal_counter
# 							coord_lower_left["y"]=vertical_counter
# 							#compute area and compare it 
# 							a=height-coord_upper_left["y"]
# 							if(a>area):
# 								area=a
# 								ymax=height-coord_upper_left["y"]
# 								ymin=height-coord_lower_left["y"]-1
# 								xmax=coord_lower_left["x"]+1
# 								xmin=coord_lower_left["x"]
# 							break
# 						#lower left coordinate has already been set
# 						#so we are basically checking vertical lines along our initial pixel line at the top
# 						if((real_image[vertical_counter,horizontal_counter]==0 and lower_left)):
# 							if(coord_lower_left["y"]<vertical_counter-1):
# 							#we went lower than the what had already ben set as a lower left bound
# 							#so a new unhindered element has to be created 
# 							#UNHINDERED RECTANGLES are those that continue to grow horizontally without meeting an obstacle
# 							#unhindered elements contain current lower left coordinates as well as upper left coordinates
# 							#they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
# 								len_unhindered=len(unhindered)
# 								#we do not want to make hindered rectangles unhindered so we have to check that they are not already set
# 								already=False
# 								for l in range(len_unhindered):
# 									if(unhindered[l][0]==coord_lower_left["y"]):
# 										already=True
# 								if(not(already)):		
# 									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
# 								#we set new lower bounds and upper bounds accordingly
# 								coord_lower_left["x"]=horizontal_counter
# 								coord_lower_left["y"]=vertical_counter-1
# 								coord_upper_left["x"]=horizontal_counter
# 								#upper counter "y" coordinate remains the same
# 								#compute the area and compare it accordignly
# 								a=vertical_counter-coord_upper_left["y"]
# 								if(a>area):
# 									area=a
# 									ymax=height-coord_upper_left["y"]
# 									ymin=height-coord_lower_left["y"]-1
# 									xmax=horizontal_counter+1
# 									xmin=coord_upper_left["x"]
# 								#Now we compute areas of above unhindered rectangles
# 								#and compare their areas
# 								length_unhindered=len(unhindered)
								
# 								for l in range(length_unhindered):
# 									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 									if(unhindered_area>area):
# 										area=unhindered_area
# 										ymax=(height-unhindered[l][2])
# 										ymin=(height-unhindered[l][0]-1)
# 										xmax=(horizontal_counter+1)
# 										xmin=unhindered[l][1]
# 								break
# 							#We remained higher than our lower left bound that we had set
# 							#we therefore get rid of unhindered rectangles that do not need 
# 							#to be considered because they have become hindered
# 							#new unhindered are also created
# 							if((coord_lower_left["y"]>(vertical_counter-1))):
# 								#first insert a new unhindered element lower bound between current hindered elements
# 								#correct particular exceptions
# 								length_unhindered=len(unhindered)
# 								checked=False
# 								added=False
# 								for l in range(length_unhindered):
# 									if(unhindered[l][0]<vertical_counter-1):
# 										checked=True
# 									if(unhindered[l][0]>vertical_counter-1):
# 										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
# 										added=True
# 										break
# 								if(checked and not(added)):
# 									unhindered.append([(vertical_counter-1),(coord_lower_left["x"]),coord_upper_left["y"],coord_lower_left["x"]])
# 								#now get rid of hindered elements
# 								length_unhindered=len(unhindered)
# 								if(length_unhindered!=0):
# 									indices=[]
# 									for l in range(length_unhindered):
# 										if(unhindered[l][0]>(vertical_counter-1)):
# 											indices.append(l)
# 									indices.reverse()
# 									for indice in indices:
# 										unhindered.pop(indice)
# 								#compute remaining areas
# 								#first check to see if there were indeed unhidered elements previously created
# 								#compute and compare their areas
# 								if(length_unhindered!=0):
# 									length_unhindered=len(unhindered)
# 									coord_lower_left["y"]=vertical_counter-1
# 									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
# 									for l in range(length_unhindered):
# 										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 										if(unhindered_area>area):
# 											area=unhindered_area
# 											ymax=(height-unhindered[l][2])
# 											ymin=(height-unhindered[l][0]-1)
# 											xmax=(horizontal_counter+1)
# 											xmin=unhindered[l][1]
										
# 								if(length_unhindered==0):
# 									coord_lower_left["y"]=vertical_counter-1
# 									#compute one area
# 									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
# 									if(a>area):
# 										area=a
# 										ymax=height-coord_upper_left["y"]
# 										ymin=height-coord_lower_left["y"]-1
# 										xmax=horizontal_counter+1
# 										xmin=coord_lower_left["x"]
# 								break
# 							#if we stay at the same lower bound
# 							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
# 								#compute and compare
# 								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
# 								if(a>area):
# 										area=a
# 										ymax=height-coord_upper_left["y"]
# 										ymin=height-coord_lower_left["y"]-1
# 										xmax=horizontal_counter+1
# 										xmin=coord_lower_left["x"]
# 								#check unhindered elements
# 								length_unhindered=len(unhindered)
# 								for l in range(length_unhindered):
# 									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 									if(unhindered_area>area):
# 										area=unhindered_area
# 										ymax=(height-unhindered[l][2])
# 										ymin=(height-unhindered[l][0]-1)
# 										xmax=(horizontal_counter+1)
# 										xmin=unhindered[l][1]
# 								break
# 							break
# 						#Special case where we hit the bottom
# 						if((real_image[vertical_counter,horizontal_counter]==1 and lower_left and vertical_counter==(height-1))):
# 							if(coord_lower_left["y"]<vertical_counter):
# 							#we went lower than the what had already ben set as a lower left bound
# 							#so a new unhindered element has to be created 
# 							#unhindered elements contain current lower left coordinates as well as upper left coordinates
# 							#they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
# 								len_unhindered=len(unhindered)
# 								already=False
# 								for l in range(len_unhindered):
# 									if(unhindered[l][0]==coord_lower_left["y"]):
# 										already=True
# 								if(not(already)):		
# 									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
# 								#we set new lower bounds and upper bounds accordingly
# 								coord_lower_left["x"]=horizontal_counter
# 								coord_lower_left["y"]=vertical_counter
# 								coord_upper_left["x"]=horizontal_counter
# 								#upper counter "y" coordinate remains the same
# 								#compute the area and compare it accordignly
# 								a=height-coord_upper_left["y"]
# 								if(a>area):
# 									area=a
# 									ymax=height-coord_upper_left["y"]
# 									ymin=height-coord_lower_left["y"]-1
# 									xmax=horizontal_counter+1
# 									xmin=coord_upper_left["x"]
# 								#Now we compute areas of above unhindered rectangles
# 								#and compare their areas
# 								length_unhindered=len(unhindered)
# 								for l in range(length_unhindered):
# 									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 									if(unhindered_area>area):
# 										area=unhindered_area
# 										ymax=(height-unhindered[l][2])
# 										ymin=(height-unhindered[l][0]-1)
# 										xmax=(horizontal_counter+1)
# 										xmin=unhindered[l][1]
# 								break
# 							#We remained higher than our lower left bound that we had set
# 							#we therefore get rid of unhindered elements (rectangles) that do not need 
# 							#to be considered because they have become hindered
# 							if((coord_lower_left["y"]>(vertical_counter))):
# 								#first insert a new unhindered element lower bound between current hindered elements
# 								length_unhindered=len(unhindered)
# 								for l in range(length_unhindered):
# 									if(unhindered[l][0]>vertical_counter-1):
# 										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
# 										break
# 								#now get rid of hindered elements
# 								length_unhindered=len(unhindered)
# 								if(length_unhindered!=0):
# 									indices=[]
# 									for l in range(length_unhindered):
# 										if(unhindered[l][0]>(vertical_counter-1)):
# 											indices.append(l)
# 									indices.reverse()
# 									for indice in indices:	
# 										unhindered.pop(indice)
# 								#compute remaining areas
# 								#first check to see if there were indeed unhidered elements previously created
# 								#compute and compare their areas
# 								if(length_unhindered!=0):
# 									length_unhindered=len(unhindered)
# 									coord_lower_left["y"]=vertical_counter-1
# 									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
# 									for l in range(length_unhindered):
# 										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 										if(unhindered_area>area):
# 											area=unhindered_area
# 											ymax=(height-unhindered[l][2])
# 											ymin=(height-unhindered[l][0]-1)
# 											xmax=(horizontal_counter+1)
# 											xmin=unhindered[l][1]
										
# 								if(length_unhindered==0):
# 									coord_lower_left["y"]=vertical_counter-1
# 									#compute one area
# 									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
# 									if(a>area):
# 										area=a
# 										ymax=height-coord_upper_left["y"]
# 										ymin=height-coord_lower_left["y"]-1
# 										xmax=horizontal_counter+1
# 										xmin=coord_lower_left["x"]
# 								break
# 							#if we stay at the same lower bound
# 							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
# 								#compute and compare
# 								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))

# 								if(a>area):
# 										area=a
# 										ymax=height-coord_upper_left["y"]
# 										ymin=height-coord_lower_left["y"]-1
# 										xmax=horizontal_counter+1
# 										xmin=coord_lower_left["x"]
# 								#check unhindered elements
# 								length_unhindered=len(unhindered)
# 								for l in range(length_unhindered):
# 									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
# 									if(unhindered_area>area):
# 										area=unhindered_area
# 										ymax=(height-unhindered[l][2])
# 										ymin=(height-unhindered[l][0]-1)
# 										xmax=(horizontal_counter+1)
# 										xmin=unhindered[l][1]				
# 								break
# 							break
# 				init_variables()
# 	end=time.time()
# 	print("the process took %lf seconds" %(end-start)) 

	

# 	results = [xmin,(height-ymax),(xmax-1),(height-ymin-1)]
# 	getinformation(results,original_img)
	
	
# 	#return results

# def getinformation(result,image_path):
# 		savePath = "/home/user/matting/area_calculate"
# 		print(result)
# 		x1,y1,x2,y2 = result[0],result[1],result[2],result[3]
# 		area = (x2-x1)*(y2-y1)
# 		print(area)
# 		center_X =x1 + int((x2-x1)/2)
# 		center_Y = y1 + int((y2-y1)/2)
# 		print(center_X)
# 		print(center_Y)
		
		
# 		aa =cv2.circle(image_path, (center_X,center_Y), 2,(255,0,0),2)
# 		bb =cv2.rectangle(aa, (x1,y1), (x2,y2),(0,255,0), thickness=1, lineType=8, shift=0)

# 		cv2.imwrite("save.jpg",bb)
# 		cv2.imshow("image",bb)

# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()
		 
















# #savefile_path = "/home/user/matting/preprocessingfile/area/"

# if __name__ == '__main__':
# 	# dataset_root_path = r"/home/user/matting/img_output/"
# 	# image_path = os.path.join(dataset_root_path,"com")
# 	# print(image_path)
# 	image_path="save2.jpg"
# 	#maxmalRectangle(image_path)
# 	# turn(image_path)
# 	find_max_rectangle(image_path)
	

	
# 	# b = getinformation(a)
# 	# #cv2.imshow("image",real_image)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()
