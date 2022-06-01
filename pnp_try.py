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
import openpyxl
import torchvision
import numpy as np
import pandas as pd

import csv
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
    init.camera_fps = 10
    init.coordinate_units = sl.UNIT.METER
    # init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE      # 深度模式  (默认-PERFORMANCE)
    # init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
    # init.depth_minimum_distance = 300
    # init.depth_maximum_distance = 5000 
    init.depth_minimum_distance = 0.2
    init.depth_maximum_distance = 2
    # init.coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 8)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 9)

    

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
    # image_size.width = 960
    # image_size.height = 540

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
        # print("1111111111111111111111111111111111")

        # cv2.imshow("left",color_image_left)
        cv2.imshow("depth",depth_image)
        # cv2.imshow("right",color_image_right)

    
        # camera_matrix_left = np.array([
        #                         (1911.0699 , 0.0 , 957.6700),
        #                         (0.0 , 1911.3900  , 573.3440),
        #                         (0.0 , 0.0 ,  1)])

        camera_matrix_left = np.array([
                                (focal_left_x  , 0.0 , center_point_x_left),
                                (0.0 , focal_left_y  , center_point_y_left),
                                (0.0 , 0.0 ,  1)])

        camera_matrix_right = np.array([
                                (1907.0100  , 0.0 , 1163.6100),
                                (0.0 , 1907.9301  , 615.6560),
                                (0.0 , 0.0 ,  1)])

        #pixel coordinate
        points_2D_left = np.array([
                        (676.0, 746.0),  #point_A
                        (754.0, 746.0),  #point_B
                        (744.0, 684.0),  #point_C
                        (674.0, 574.0),  #point_D
                        (750.0, 574.0),  #point_E
                        (762.0, 530.0)  #point_F
                      ], dtype="double")

        points_2D_right = np.array([
                        (424.0, 238.0),  #point_A
                        (405.0, 201.0),  #point_B
                        (466.0, 242.0),  #point_C
                        (489.0, 207.0),  #point_D
                        (429.0, 167.0),  #point_E
                        (471.0, 168.0)  #point_F
                      ], dtype="double")
        
        #world coordinate
        points_3D = np.array([
                    (0.02, 0, 0.02),  #point_A
                    (0.04, 0, 0.02),  #point_B
                    (0.04, 0.0, 0.04),  #point_C
                    (0.02, 0.02 ,0.06),  #point_D
                    (0.04, 0.02, 0.06),  #point_E
                    (0.04, 0.04, 0.06)  #point_F
                    ])

        # points_3D = np.array([
        #               (0.0, 0.02, 0.02),  #point_A
        #               (0.0, 0.04, 0.04),  #point_B
        #               (0.02, 0.0, 0.02),  #point_C
        #               (0.04, 0.0, 0.04),  #point_D
        #               (0.02, 0.04, 0.06),  #point_E
        #               (0.04, 0.02, 0.06)  #point_F
        #              ])


        dist_coeffs = np.zeros((5,1))
        # dist_coeffs_left  = np.mat([-0.059,-0.0655,-0.0007,-0.0006,0.000])


        success, rotation_vector_left, translation_vector_left = cv2.solvePnP(points_3D, points_2D_left, camera_matrix_left, dist_coeffs, flags=0)

        success, rotation_vector_right, translation_vector_right = cv2.solvePnP(points_3D, points_2D_right, camera_matrix_right, dist_coeffs, flags=0)

        print("rotation_vector_left:")
        print(rotation_vector_left)
        print("translation_vector_left:")
        print(translation_vector_left)

        
        #旋转向量转换旋转矩阵
        rotation_matrix_left = cv2.Rodrigues(rotation_vector_left)[0]
        rotation_matrix_right = cv2.Rodrigues(rotation_vector_right)[0]

        
        RTmatrix_left = np.hstack((rotation_matrix_left, translation_vector_left))

        RTmatrix_right = np.hstack((rotation_matrix_right, translation_vector_right))
        print("RTmatrix RTmatrix_left: ")
        print(rotation_matrix_left)

        # rint("RTmatrix RTmatrix_left: ")
        print(RTmatrix_left)

        # print("RTmatrix right: ")
        # print(RTmatrix_right)

        array1=np.array([[0, 0,0,1]])
        new_rt = np.vstack((RTmatrix_left,array1))
        print("new_rt")
        print(new_rt)


        





        # d = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}s
        df = pd.DataFrame(data=RTmatrix_left)
        df.columns=['X','Y','Z','t']

        # da = pd.DataFrame(data=RTmatrix_right)
        # da.columns=['X','Y','Z','t']

        # dz = pd.DataFrame()
        # dz = pd.concat([df,da],axis=0)
        # dz = pd.merge(df, da, on="X")
        df.to_excel("test2.xlsx")
        df.to_csv("test2.csv")
        # dz.to_csv("test2.csv")
        # da.to_csv("test2.csv")


        #3d to 2d
        world = np.array([0.02, 0, 0.02,1]).T.reshape(-1, 1)
        print(world.shape)
        
        # print(camera_matrix.shape, RTmatrix.shape, world.shape)
        # aa = camera_matrix_left.dot(RTmatrix_left.dot)# * world
        # aa = aa.dot(world)
        # aa /= aa[-1]
        pc = new_rt.dot(world)
        
        print("pixel cooridinate:")
        print(pc)                                                      
        new_cl = np.hstack((camera_matrix_left, np.zeros((3,1))))
        pi = new_cl.dot(pc)
        pi /= pi[-1]
        print(pi)
        # exs
        # color_image_left_viz = cv2.resize(color_image_left, (1920//2, 1080//2))
        # cv2.circle(color_image_left, (int(aa[0]), int(aa[1])), 2, (0,0,255), -1)
        # cv2.imshow("3d to 2d",color_image_left)
        # cv2.imshow("1111",color_image_left)

        point3D_1 = point_cloud.get_value(688,530)
        x1 = point3D_1[1][0]
        y1 = point3D_1[1][1]
        z1 = point3D_1[1][2]

        point3D_2= point_cloud.get_value(688,220)
        x2 = point3D_2[1][0]
        y2 = point3D_2[1][1]
        z2 = point3D_2[1][2]

        point3D_3= point_cloud.get_value(579,220)
        x3 = point3D_3[1][0]
        y3 = point3D_3[1][1]
        z3 = point3D_3[1][2]

        point3D_4= point_cloud.get_value(578,277)
        x4 = point3D_4[1][0]
        y4 = point3D_4[1][1]
        z4 = point3D_4[1][2]

        point3D_point = point_cloud.get_value(465,279)
        point_x = point3D_point[1][0]
        point_y = point3D_point[1][1]
        point_z = point3D_point[1][2]

        
        # print("point3D_point: "+ str(A_x)+ " ," + str(A_y) + " ," + str(A_z))

        print("point3D_1: "+ str(x1)+ " ," + str(y1) + " ," + str(z1))
        # print("point3D_2: "+ str(x2)+ " ," + str(y2) + " ," + str(z2))
        # print("point3D_3: "+ str(x3)+ " ," + str(y3) + " ," + str(z3))
        # print("point3D_4: "+ str(x4)+ " ," + str(y4) + " ," + str(z4))
        # print("point3D_down: "+ str(point_x)+ " ," + str(point_y) + " ," + str(point_z))


        z_value= depth_image_zed.get_value(1096, 456)
        # cv2.circle(color_image_left, (int(676*2), int(276*2)), 2, (0,0,255), -1)

        testx = (1096 - center_point_x_left)* z_value[1] / focal_left_x
        testy = (456 - center_point_y_left) * z_value[1] / focal_left_y
        testz = z_value[1]
        print("cam_point:" +str(testx) + "," + str(testy) + ","+ str(testz) )
        
        # RTmatrix1_inv = np.linalg.inv(rotation_matrix_left)

        c_1=np.array([[testx, testy,testz,1]])
        c11_1=c_1.T


        new1_rt = np.linalg.inv(new_rt)
        ans = np.dot(new1_rt,c11_1)
        print("ans:")
        print(ans)
        
        
        # ans1 = c_1 + translation_vector_left
        # ans = RTmatrix1_inv.dot(ans1)
        # print("ans1:")
        # print(ans)


        # dis  = abs(point_y - A_y)
        # print("dis: " + str(dis))

        # print("--------------------")
        

        #2d to 3d
        RTmatrix1_inv = np.linalg.inv(rotation_matrix_left)
        inverse_newcam_mtx = np.linalg.inv(camera_matrix_left)
        # print(inverse_newcam_mtx)

        # uv_1=np.array([[531,290,1]], dtype=np.float32)
        uv_1=np.array([[1096,456,1]])
        uv_1=uv_1.T

        invR_x_invM_x_uv1 = np.dot(np.dot(RTmatrix1_inv,inverse_newcam_mtx), uv_1)
        invR_x_tvec = np.dot(RTmatrix1_inv, translation_vector_left)
        Z_left =  0
        #  Z_left = dis - invR_x_tvec[2]/invR_x_invM_x_uv1[2]
        output = (Z_left+invR_x_tvec[2, 0])/invR_x_invM_x_uv1[2, 0]*invR_x_invM_x_uv1-invR_x_tvec
        print(output)

        # RTmatrix1_inv = np.linalg.inv(rotation_matrix_left)
        # inverse_newcam_mtx = np.linalg.inv(camera_matrix_left)


        # uv_1=np.array([[688,530,1]])
        # uv_1=uv_1.T 
       
        # mt = rotation_matrix_left.dot(translation_vector_left)
        # uv = 0.48893532156944275 * uv_1
        # # uv = uv - mt
        # aa = inverse_newcam_mtx.dot(RTmatrix1_inv)
        # tt = aa.dot(uv)
        # cc = tt - mt
        # print(cc)


        # #2d to 3d
        # RTmatrix1_inv_right = np.linalg.inv(rotation_matrix_right)
        # inverse_newcam_mtx_right = np.linalg.inv(camera_matrix_right)
        # # print(inverse_newcam_mtx)

        # # uv_1=np.array([[531,290,1]], dtype=np.float32)
        # uv_1_right=np.array([[298,388,1]])
        # uv_1_right=uv_1_right.T

        # invR_x_invM_x_uv1_right = np.dot(np.dot(RTmatrix1_inv_right,inverse_newcam_mtx_right), uv_1_right)
        # invR_x_tvec_right = np.dot(RTmatrix1_inv_right, translation_vector_right)

        # Z_right = 0.018
        # # print(z_right)
        # output_right = (Z_right+invR_x_tvec_right[2, 0])/invR_x_invM_x_uv1_right[2, 0]*invR_x_invM_x_uv1_right-invR_x_tvec_right
        # print("out_right")
        # print(output_right)

       
        cv2.imshow("1111",color_image_left)

        key = cv2.waitKey(5)
        if key == 27:
            break  
        
