from tkinter import *
import cv2
import tkinter as tk
import logging
from PIL import Image,ImageTk
import pyzed.sl as sl
import numpy as np
import datetime
import ogl_viewer.viewer as gl

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_pose(pose, tx) :
    transform_ = sl.Transform()
    transform_.set_identity()
    # Translate the tracking frame by tx along the X axis
    transform_[0][3] = tx
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform_)
    transform_inv.inverse()
    pose = transform_inv * pose * transform_





if __name__ == '__main__': 
    #定義zed
    zed = sl.Camera()
    
    zed_pose = sl.Pose()

    objects = sl.Objects()
    
    #init
    input_type = sl.InputType() # Set configuration parameters
    input_type.set_from_camera_id(0)
    init = sl.InitParameters(input_t=input_type)  # 初始化
    init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
    init.camera_fps = 15
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True
    # init.coordinate_units = sl.UNIT.MILLIMETER

    init.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
    init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
    # init.depth_minimum_distance = 300
    # init.depth_maximum_distance = 5000 
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 5 

    

    #open camera
    if not zed.is_opened():
        log.info("Opening ZED Camera...")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    #set zed runtime value
    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL


    #set image size
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = 960
    image_size.height = 540

    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(len(sys.argv), sys.argv,calibration_params,image_size)

    #turn zed to numpy(for opencv)
    image_zed_left = sl.Mat(image_size.width, image_size.height)
    image_zed_right = sl.Mat(image_size.width, image_size.height)
    depth_image_zed = sl.Mat(image_size.width,image_size.height)
    point_cloud = sl.Mat(image_size.width,image_size.height)
    # point_cloud1 = sl.Mat(image_size.width,image_size.height,sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    # point_cloud = sl.Mat()
    

    # camera_disable_self_calib = False
    center =np.array([571,520,546])


    key = ''
    while key != 113:
        zed.grab() #開啟管道
        zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        # zed.retrieve_measure(point_cloud1, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        # print(point_cloud1)
        # zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)
        # color_image = image_zed_left.get_data()
        # color_image_right = image_zed_right.get_data()
        # depth_image = depth_image_zed.get_data()
        
        # viewer.updateData(point_cloud1)

        point3D = point_cloud.get_value(571,520)
        x = point3D[1][0]
        y = point3D[1][1]
        z = point3D[1][2]
        color = point3D[1][3]
        # print(x,y,z)

        calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        focal_left_x = calibration_params.left_cam.fx
        focal_left_y = calibration_params.left_cam.fy
        focal_right_x = calibration_params.right_cam.fx
        focal_right_y = calibration_params.right_cam.fy
        center_point_x = calibration_params.left_cam.cx
        center_point_y = calibration_params.left_cam.cy
        translate = calibration_params.T
        rotation = calibration_params.R
        py_Rotation = sl.Rotation()
        new = py_Rotation.set_rotation_vector(rotation[0],rotation[1],rotation[2])


        print(focal_left_x,focal_left_y)
        print(focal_right_x,focal_right_y)
        print(center_point_x,center_point_y)
        print(translate)
        print(rotation)
        print(new)


        
        # print(t)
        # print("transform: tx: {0}, ty: {1}, tz {2}\n".format(tx, ty, tz))
        # # Retrieve and transform the pose data into a new frame located at the center of the camera
        # tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        # transform_pose(zed_pose.pose_data(sl.Transform()), translation_left_to_center)
        # print(transform_pose)


        cv2.waitKey(0)     
        if key == 27:
            break

    zed.close()  
    viewer.exit()
    cv2.destroyAllWindows()
    
