import os
import sys
import cv2
import uuid
import glob
import time
import math
import shutil
import random
import torch
import darknet
import logging
import shutil
import datetime
import argparse
import torchvision
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl

from pylsd.lsd import lsd
from ctypes import *
# from random import randintre
from tqdm import tqdm
from PIL import Image
from queue import Queue
from shutil import copyfile
from threading import Thread, enumerate 

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


zed_id = 0
fx = 1949.55078125
fy = 1949.55078125
cx = 989.0401611328125
cy = 562.3345947265625

def calculate_XYZ(self,u,v):
                                      
        #Solve: From Image Pixels, find World Points

        uv_1=np.array([[u,v,1]], dtype=np.float32)
        uv_1=uv_1.T
        suv_1=self.scalingfactor*uv_1
        xyz_c=self.inverse_newcam_mtx.dot(suv_1)
        xyz_c=xyz_c-self.tvec1
        XYZ=self.inverse_R_mtx.dot(xyz_c)

        return XYZ

if __name__ == '__main__':
    #定義zed
    zed = sl.Camera()
    zed_pose = sl.Pose()

    #init
    input_type = sl.InputType() # Set configuration parameters
    input_type.set_from_camera_id(zed_id)
    init = sl.InitParameters(input_t=input_type)  # 初始化
    init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
    init.camera_fps = 15
    init.coordinate_units = sl.UNIT.METER
    # init.coordinate_units = sl.UNIT.MILLIMETER
    # init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
    init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
    # init.depth_minimum_distance = 300
    # init.depth_maximum_distance = 5000 
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 5 
    init.coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    

     #open camera
    if not zed.is_opened():
        log.info("Opening ZED Camera...")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

    #set image size
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = 960
    image_size.height = 540

    image_zed_left = sl.Mat(image_size.width, image_size.height)
    image_zed_right = sl.Mat(image_size.width, image_size.height)
    depth_image_zed = sl.Mat(image_size.width,image_size.height)
    point_cloud = sl.Mat(image_size.width,image_size.height)
    point_cloud1 = sl.Mat(image_size.width,image_size.height)

    key = ''
    while key != 113 : 
        zed.grab() #開啟管道
        zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        zed.retrieve_measure(point_cloud1, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        
        color_image_left = image_zed_left.get_data()
        color_image_right = image_zed_right.get_data()
        depth_image = depth_image_zed.get_data()
        # print(np.min(depth_image), np.max(depth_image))

        #camera parameter
        calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        focal_left_x = calibration_params.left_cam.fx
        focal_left_y = calibration_params.left_cam.fy

        focal_right_x = calibration_params.right_cam.fx
        focal_right_y = calibration_params.right_cam.fy

        center_point_x_left = calibration_params.left_cam.cx
        center_point_y_left = calibration_params.left_cam.cy

        center_point_x_right = calibration_params.right_cam.cx
        center_point_y_right = calibration_params.right_cam.cy

        translate = calibration_params.T
        rotation = calibration_params.R

        print(focal_left_x,focal_left_y)
        print(focal_right_x,focal_right_x)
        print(center_point_x_left,center_point_y_left)
        print(center_point_x_right,center_point_y_right)
        print("1111111111111111111111111111111111")

        # cv2.imshow("left",color_image_left)
        # cv2.imshow("right",color_image_right)

        

        #camera matrix
        camera_matrix_left = np.array([
                                (1904.5601  , 0.0 , 949.1600),
                                (0.0 , 1904.8108  , 554.9330),
                                (0.0 , 0.0 ,  1)])

        camera_matrix_right = np.array([
                                (1916.4301  , 0.0 , 1016.04),
                                (0.0 , 1917.03  , 566.847),
                                (0.0 , 0.0 ,  1)])

        #pixel coordinate
        points_2D_left = np.array([
                        (727.0, 295.0),  #point_A
                        (702.0, 256.0),  #point_B
                        (775.0, 293.0),  #point_C
                        (795.0, 252.0),  #point_D
                        (724.0, 216.0),  #point_E
                        (770.0, 199.0)  #point_F
                      ], dtype="double")

        points_2D_right = np.array([
                        (526.0, 295.0),  #point_A
                        (501.0, 256.0),  #point_B
                        (573.0, 293.0),  #point_C
                        (594.0, 252.0),  #point_D
                        (523.0, 216.0),  #point_E
                        (569.0, 214.0)  #point_F
                      ], dtype="double")
        
        #world coordinate
        points_3D = np.array([
                      (0.0, 0.02, 0.02),  #point_A
                      (0.0, 0.04, 0.04),  #point_B
                      (0.02, 0.0, 0.02),  #point_C
                      (0.04, 0.0, 0.04),  #point_D
                      (0.02, 0.04, 0.06),  #point_E
                      (0.04, 0.02, 0.06)  #point_F
                     ])

        dist_coeffs = np.zeros((5,1))
        # dist_coeffs  = np.mat([-0.0593077,-0.0652694,0.000604454,0.000399874,0])

        success, rotation_vector_left, translation_vector_left = cv2.solvePnP(points_3D, points_2D_left, camera_matrix_left, dist_coeffs, flags=0)

        success, rotation_vector_right, translation_vector_right = cv2.solvePnP(points_3D, points_2D_right, camera_matrix_right, dist_coeffs, flags=0)

        #旋转向量转换旋转矩阵
        rotation_matrix_left = cv2.Rodrigues(rotation_vector_left)[0]
        rotation_matrix_right = cv2.Rodrigues(rotation_vector_right)[0]
        
        RTmatrix_left = np.hstack((rotation_matrix_left, translation_vector_left))
        RTmatrix_right = np.hstack((rotation_matrix_right, translation_vector_right))
        print("RTmatrix left: ")
        print(RTmatrix_left)

        print("RTmatrix right: ")
        print(RTmatrix_right)
        

        # #3d to 2d
        world = np.array([0,0.04,0.04,1]).T.reshape(-1, 1)
        # print(camera_matrix.shape, RTmatrix.shape, world.shape)
        aa = camera_matrix_left.dot(RTmatrix_left)# * world
        aa = aa.dot(world)
        aa /= aa[-1]
        print("pixel cooridinate:")
        print(aa)
        cv2.circle(color_image_left, (int(aa[0]), int(aa[1])), 2, (0,0,255), -1)

        point3D_down = point_cloud.get_value(537,305)
        A_x = point3D_down[1][0]
        A_y = point3D_down[1][1]
        A_z = point3D_down[1][2]

        point3D_point = point_cloud.get_value(538,250)
        point_x = point3D_point[1][0]
        point_y = point3D_point[1][1]
        point_z = point3D_point[1][2]

        print("point3D_down: "+ str(A_x)+ " ," + str(A_y) + " ," + str(A_z))
        print("point3D_point: "+ str(point_x)+ " ," + str(point_y) + " ," + str(point_z))

        dis  = abs(point_y - A_y)
        print("dis: " + str(dis))
        print("--------------------")



        #2d to 3d
        # RTmatrix1_inv = np.linalg.inv(rotation_matrix_left)
        # inverse_newcam_mtx = np.linalg.inv(camera_matrix_left)
        # # print(inverse_newcam_mtx)

        # # uv_1=np.array([[531,290,1]], dtype=np.float32)
        # uv_1=np.array([[282.0,264.0,1]])
        # uv_1=uv_1.T

        # invR_x_invM_x_uv1 = np.dot(np.dot(RTmatrix1_inv,inverse_newcam_mtx), uv_1)
        # invR_x_tvec = np.dot(RTmatrix1_inv, translation_vector_left)
        # Z = 0.022347
        # output = (Z+invR_x_tvec[2, 0])/invR_x_invM_x_uv1[2, 0]*invR_x_invM_x_uv1-invR_x_tvec
        # print(output)

        #2d to 3d
        RTmatrix1_inv_right = np.linalg.inv(rotation_matrix_right)
        inverse_newcam_mtx_right = np.linalg.inv(camera_matrix_right)
        # print(inverse_newcam_mtx)

        # uv_1=np.array([[531,290,1]], dtype=np.float32)
        uv_1=np.array([[327.0,189.0,1]])
        uv_1=uv_1.T

        invR_x_invM_x_uv1 = np.dot(np.dot(RTmatrix1_inv_right,inverse_newcam_mtx_right), uv_1)
        invR_x_tvec = np.dot(RTmatrix1_inv_right, translation_vector_right)
        Z = 0.0602
        output = (Z+invR_x_tvec[2, 0])/invR_x_invM_x_uv1[2, 0]*invR_x_invM_x_uv1-invR_x_tvec
        print(output)

       
        # cv2.imshow("1111",color_image_left)

        key = cv2.waitKey(5)
        if key == 27:
            break  
        
