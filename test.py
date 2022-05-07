from tkinter import *
import cv2
import tkinter as tk
import logging
from PIL import Image,ImageTk
import pyzed.sl as sl
import numpy as np
import datetime

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

global current_image

def take_shot():
    flag = 1
    print(flag)
    image = Image.fromarray(current_image)
    time = str(datetime.datetime.now().today()).replace(":"," ") + ".jpg"
    image.save(time)
    imgtk = tk.PhotoImage(file = "1111.gif")
    imgLabel = tk.Label(window,image=imgtk)
    imgLabel.place(x=1200,y=50)#自动对齐

#     photo = tk.PhotoImage(file="18.png")#file：t图片路径
# imgLabel = tk.Label(root,image=photo)


    # imgtk = ImageTk.PhotoImage(image=current_image)

    # current_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)#
    # imgtk = ImageTk.PhotoImage(Image.fromarray(current_image))
    

    
    
    # current_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)#
    # current_image = Image.fromarray(current_image)#将图像转换成Image对象
    # imgtk = ImageTk.PhotoImage(image=current_image)
    # panel1.imgtk = imgtk
    # panel1.config(image=imgtk)

# def video_loop():
    

#     if flag == 1:
#         current_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)#
#         current_image = Image.fromarray(current_image)#将图像转换成Image对象
#         imgtk = ImageTk.PhotoImage(image=current_image)
        
#         show_image.configure(image=imgtk)
#         show_image.image = imgtk

#     window.after(5, video_loop)
#     cv2.imshow("color_image",color_image)


if __name__ == '__main__':
    global flag
    flag = 0
    #GUI
    window = Tk()
    window.title("yoyoyin")
    window.geometry('1920x1070')
    window.resizable(0,0)   
    window.config(cursor="heart") #只是游標

    RGBCamera = Label(window,relief=SUNKEN)  # initialize image panel
    RGBCamera.place(x=0,y=0)

    DepthCamera = Label(window,relief=SUNKEN)  # initialize image panel
    DepthCamera.place(x=0,y=540)

    panel3 = LabelFrame(window, text="Operation Result",fg='#6A6AFF',font=('Arial',18),bg='#FFECF5',width=955,height=1070,relief=SUNKEN)
    panel3.place(x=962,y=0)

    show_image = Label(window,relief=SUNKEN)
    show_image.place(x=1200,y=50)

    btn1 = Button(window, text="take Snapshot!",bg='#CECEFF',fg='#6A6AFF',font=('Arial',12), command=take_shot)
    btn1['width'] = 40
    btn1['height'] = 4
    # btn.grid(column=0,row=0)
    btn1.place(x=1000,y=600)
    
    #定義zed
    zed = sl.Camera()

    #init
    input_type = sl.InputType() # Set configuration parameters
    input_type.set_from_camera_id(0)
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

    #turn zed to numpy(for opencv)
    image_zed_left = sl.Mat(image_size.width, image_size.height)
    image_zed_right = sl.Mat(image_size.width, image_size.height)
    depth_image_zed = sl.Mat(image_size.width,image_size.height)
    point_cloud = sl.Mat(image_size.width,image_size.height)

    while True:
        zed.grab() #開啟管道
        zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
        # zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)
        color_image = image_zed_left.get_data()
        color_image_right = image_zed_right.get_data()
        depth_image = depth_image_zed.get_data()
        # print(depth_image.shape)
        # print(np.max(depth_image))
        disp_depth_image = cv2.normalize(depth_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
        disp_depth_image = cv2.applyColorMap(disp_depth_image, cv2.COLORMAP_JET)

        #GUI顯示Cam畫面
        current_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)#
        imgtk = ImageTk.PhotoImage(Image.fromarray(current_image))
        RGBCamera['image'] = imgtk
        # RGBCamera.config(image=imgtk)

        disp_depth_image = cv2.cvtColor(disp_depth_image, cv2.COLOR_BGR2RGB)#
        depthtk = ImageTk.PhotoImage(Image.fromarray(disp_depth_image))
        
        DepthCamera['image']= depthtk
        # DepthCamera.config(image=depthtk)

        window.update()

    window.mainloop()
    # 当一切都完成后，关闭摄像头并释放所占资源
    # camera.release()
    cv2.destroyAllWindows()







# from tkinter import *
# import cv2
# from PIL import Image,ImageTk
# import pyzed.sl as sl
# import logging

# # Get the top-level logger object
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# def take_snapshot():
#     print("有人给你点赞啦！")

# def video_loop():
#     zed = sl.Camera()
#     #init
#     input_type = sl.InputType() # Set configuration parameters
#     input_type.set_from_camera_id(0)
#     init = sl.InitParameters(input_t=input_type)  # 初始化
#     init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
#     init.camera_fps = 15
#     init.coordinate_units = sl.UNIT.METER
#     # init.coordinate_units = sl.UNIT.MILLIMETER

#     # init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
#     init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
#     # init.depth_minimum_distance = 300
#     # init.depth_maximum_distance = 5000 
#     init.depth_minimum_distance = 0.3
#     init.depth_maximum_distance = 5 

#     #open camera
#     if not zed.is_opened():
#         log.info("Opening ZED Camera...")
#     status = zed.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         log.error(repr(status))
#         exit()
    
    
#     # #set image size
#     image_size = zed.get_camera_information().camera_resolution
#     image_size.width = 600
#     image_size.height = 400

#     zed.grab()

#     #turn zed to numpy(for opencv)
#     image_zed_left = sl.Mat(image_size.width, image_size.height)
#     image_zed_right = sl.Mat(image_size.width, image_size.height)
#     depth_image_zed = sl.Mat(image_size.width,image_size.height)
#     point_cloud = sl.Mat(image_size.width,image_size.height)

#     zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
#     zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
#     zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
#     zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)

#     color_image = image_zed_left.get_data()
#     # image_right = image_zed_left.get_data()
#     depth_image = depth_image_zed.get_data()
#     # print(np.min(depth_image), np.max(depth_image))
#     cv2image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
#     current_image = Image.fromarray(cv2image)#将图像转换成Image对象
#     imgtk = ImageTk.PhotoImage(image=current_image)
#     panel.imgtk = imgtk
#     panel.config(image=imgtk)
#     root.after(1, video_loop)



# root = Tk()
# root.title("opencv + tkinter")
# #root.protocol('WM_DELETE_WINDOW', detector)

# panel = Label(root)  # initialize image panel
# panel.pack(padx=10, pady=10)
# root.config(cursor="arrow")
# btn = Button(root, text="点赞!", command=take_snapshot)
# btn.pack(fill="both", expand=True, padx=10, pady=10)

# video_loop()

# root.mainloop()
# window.mainloop()
# # 当一切都完成后，关闭摄像头并释放所占资源
# camera.release()
# cv2.destroyAllWindows()





# import tkinter as tk
# import pyzed.sl as sl
# import cv2
# import logging
# from PIL import Image, ImageTk

# # Get the top-level logger object
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# def camera_stream():
#     zed = sl.Camera()

#     #init
#     input_type = sl.InputType() # Set configuration parameters
#     input_type.set_from_camera_id(0)
#     init = sl.InitParameters(input_t=input_type)  # 初始化
#     init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
#     init.camera_fps = 15
#     init.coordinate_units = sl.UNIT.METER
#     # init.coordinate_units = sl.UNIT.MILLIMETER

#     # init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
#     init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
#     # init.depth_minimum_distance = 300
#     # init.depth_maximum_distance = 5000 
#     init.depth_minimum_distance = 0.3
#     init.depth_maximum_distance = 5 

#     #open camera
#     if not zed.is_opened():
#         log.info("Opening ZED Camera...")
#     status = zed.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         log.error(repr(status))
#         exit()
    
#     #set image size
#     image_size = zed.get_camera_information().camera_resolution
#     image_size.width = 960
#     image_size.height = 540
    
#     #set zed runtime value
#     runtime_parameters =sl.RuntimeParameters()
#     runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

#     #turn zed to numpy(for opencv)
#     image_zed_left = sl.Mat(image_size.width, image_size.height)
#     image_zed_right = sl.Mat(image_size.width, image_size.height)
#     depth_image_zed = sl.Mat(image_size.width,image_size.height)
#     point_cloud = sl.Mat(image_size.width,image_size.height)

#     zed.grab() #開啟管道

#     zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
#     zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
#     zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
#     zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)

#     color_image = image_zed_left.get_data()
#     # image_right = image_zed_left.get_data()
#     depth_image = depth_image_zed.get_data()
#     # print(np.min(depth_image), np.max(depth_image))
#     cv2image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(color_image)
#     imgtk = ImageTk.PhotoImage(image = img)
#     video.imgtk = imgtk
#     video.configure(image = imgtk)
#     video.after(1,camera_stream)

# root = tk.Tk()
# videoFrame = tk.Frame(root,bg="white").pack()
# video = tk.Label(videoFrame)
# video.pack()
# camera_stream()
# root.mainloop()
# window.mainloop()

# def define_layout(obj, cols=1, rows=1):

#     def method(trg, col, row):

#         for c in range(cols):
#             trg.columnconfigure(c, weight=1)
#         for r in range(rows):
#             trg.rowconfigure(r, weight=1)

#     if type(obj)==list:
#         [ method(trg, cols, rows) for trg in obj ]
#     else:
#         trg = obj
#         method(trg, cols, rows)


# window = tk.Tk()
# window.title('yoyoyin')
# window.geometry('1920x1080')
# align_mode = 'nswe'
# pad = 5

# div_size = 200
# img_size = div_size * 2

# div1 = tk.Frame(window,  width=960 , height=540 , bg='blue')
# div2 = tk.Frame(window,  width=960 , height=540 , bg='orange')
# div3 = tk.Frame(window,  width=960 , height=540 , bg='red')
# div3 = tk.Frame(window,  width=960 , height=540 , bg='green')

# window.update()
# win_size = min(window.winfo_width(), window.winfo_height())
# print(window.winfo_width(),window.winfo_height())
# print(win_size)
# # rowspan=2,
# div1.grid(column=0, row=0, padx=pad, pady=pad,  sticky=align_mode)
# # div2.grid(column=0, row=1, padx=pad, pady=pad, sticky=align_mode)
# # div3.grid(column=1, row=1, padx=pad, pady=pad, sticky=align_mode)

# define_layout(window, cols=2, rows=2)
# define_layout([div1, div2, div3])

# im = Image.open('/home/user/matting/long_4.jpg')
# imTK = ImageTk.PhotoImage(im)
# # imTK = ImageTk.PhotoImage( im.resize( (img_size, img_size) ) )

# image_main = tk.Label(div1, image=imTK)
# image_main['height'] = 1280
# image_main['width'] = 720

# image_main.grid(column=0, row=0, sticky=align_mode)

# lbl_title1 = tk.Label(div2, text='Hello', bg='orange', fg='white')
# lbl_title2 = tk.Label(div2, text="World", bg='orange', fg='white')

# lbl_title1.grid(column=0, row=0, sticky=align_mode)
# lbl_title2.grid(column=0, row=1, sticky=align_mode)

# bt1 = tk.Button(div3, text='Button 1', bg='pink', fg='white')
# bt1['width'] = 50
# bt1['height'] = 4
# bt1['activebackground'] = 'red'


# bt2 = tk.Button(div3, text='Button 2', bg='green', fg='white')
# bt3 = tk.Button(div3, text='Button 3', bg='green', fg='white')
# bt4 = tk.Button(div3, text='Button 4', bg='green', fg='white')

# bt1.grid(column=0, row=0)
# bt2.grid(column=0, row=1, sticky=align_mode)
# bt3.grid(column=0, row=2, sticky=align_mode)
# bt4.grid(column=0, row=3, sticky=align_mode)

# bt1['command'] = lambda : get_size(window, image_main, im)

# define_layout(window, cols=2, rows=2)
# define_layout(div1)
# define_layout(div2, rows=2)
# define_layout(div3, rows=4)

# window.mainloop()



#coding: utf-8
# import cv2
# import numpy as np
# import imutils
 
 
# img = cv2.imread('long_4.jpg')
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(imgray, 70, 210)
 
# # ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 大津阈值
# contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
 
# # cnts = contours[0] if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是2+
 
# for cnt in contours:
#     # # 外接矩形框，没有方向角
#     # x, y, w, h = cv2.boundingRect(cnt)
#     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
#     # 最小外接矩形框，有方向角
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
 
#     # 最小外接圆
#     # (x, y), radius = cv2.minEnclosingCircle(cnt)
#     # center = (int(x), int(y))
#     # radius = int(radius)
#     # cv2.circle(img, center, radius, (255, 0, 0), 2)
 
#     # 椭圆拟合
#     # ellipse = cv2.fitEllipse(cnt)
#     # cv2.ellipse(img, ellipse, (255, 255, 0), 2)
 
#     # 直线拟合
#     rows, cols = img.shape[:2]
#     [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
#     lefty = int((-x * vy / vx) + y)
#     righty = int(((cols - x) * vy / vx) + y)
#     img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
 
# cv2.imshow('a',img)
# cv2.imwrite('./result.jpg',img)
# cv2.waitKey(0)




# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2021/11/14 21:24
# # @Author  : 剑客阿良_ALiang
# # @Site    : 
# # @File    : inferance_hy.py
# import argparse
# import torch
# import os
 
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms as T
# from torchvision.transforms.functional import to_pil_image
# from threading import Thread
# from tqdm import tqdm
# from torch.utils.data import Dataset
# from PIL import Image
# from typing import Callable, Optional, List, Tuple
# import glob
# from torch import nn
# from torchvision.models.resnet import ResNet, Bottleneck
# from torch import Tensor
# import torchvision
# import numpy as np
# import cv2
# import uuid
 
 
# # --------------- hy ---------------
# class HomographicAlignment:
#     """
#     Apply homographic alignment on background to match with the source image.
#     """
 
#     def __init__(self):
#         self.detector = cv2.ORB_create()
#         self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
 
#     def __call__(self, src, bgr):
#         src = np.asarray(src)
#         bgr = np.asarray(bgr)
 
#         keypoints_src, descriptors_src = self.detector.detectAndCompute(src, None)
#         keypoints_bgr, descriptors_bgr = self.detector.detectAndCompute(bgr, None)
 
#         matches = self.matcher.match(descriptors_bgr, descriptors_src, None)
#         matches.sort(key=lambda x: x.distance, reverse=False)
#         num_good_matches = int(len(matches) * 0.15)
#         matches = matches[:num_good_matches]
 
#         points_src = np.zeros((len(matches), 2), dtype=np.float32)
#         points_bgr = np.zeros((len(matches), 2), dtype=np.float32)
#         for i, match in enumerate(matches):
#             points_src[i, :] = keypoints_src[match.trainIdx].pt
#             points_bgr[i, :] = keypoints_bgr[match.queryIdx].pt
 
#         H, _ = cv2.findHomography(points_bgr, points_src, cv2.RANSAC)
 
#         h, w = src.shape[:2]
#         bgr = cv2.warpPerspective(bgr, H, (w, h))
#         msk = cv2.warpPerspective(np.ones((h, w)), H, (w, h))
 
#         # For areas that is outside of the background,
#         # We just copy pixels from the source.
#         bgr[msk != 1] = src[msk != 1]
 
#         src = Image.fromarray(src)
#         bgr = Image.fromarray(bgr)
 
#         return src, bgr
 
 
# class Refiner(nn.Module):
#     # For TorchScript export optimization.
#     __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']
 
#     def __init__(self,
#                  mode: str,
#                  sample_pixels: int,
#                  threshold: float,
#                  kernel_size: int = 3,
#                  prevent_oversampling: bool = True,
#                  patch_crop_method: str = 'unfold',
#                  patch_replace_method: str = 'scatter_nd'):
#         super().__init__()
#         assert mode in ['full', 'sampling', 'thresholding']
#         assert kernel_size in [1, 3]
#         assert patch_crop_method in ['unfold', 'roi_align', 'gather']
#         assert patch_replace_method in ['scatter_nd', 'scatter_element']
 
#         self.mode = mode
#         self.sample_pixels = sample_pixels
#         self.threshold = threshold
#         self.kernel_size = kernel_size
#         self.prevent_oversampling = prevent_oversampling
#         self.patch_crop_method = patch_crop_method
#         self.patch_replace_method = patch_replace_method
 
#         channels = [32, 24, 16, 12, 4]
#         self.conv1 = nn.Conv2d(channels[0] + 6 + 4, channels[1], kernel_size, bias=False)
#         self.bn1 = nn.BatchNorm2d(channels[1])
#         self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels[2])
#         self.conv3 = nn.Conv2d(channels[2] + 6, channels[3], kernel_size, bias=False)
#         self.bn3 = nn.BatchNorm2d(channels[3])
#         self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size, bias=True)
#         self.relu = nn.ReLU(True)
 
#     def forward(self,
#                 src: torch.Tensor,
#                 bgr: torch.Tensor,
#                 pha: torch.Tensor,
#                 fgr: torch.Tensor,
#                 err: torch.Tensor,
#                 hid: torch.Tensor):
#         H_full, W_full = src.shape[2:]
#         H_half, W_half = H_full // 2, W_full // 2
#         H_quat, W_quat = H_full // 4, W_full // 4
 
#         src_bgr = torch.cat([src, bgr], dim=1)
 
#         if self.mode != 'full':
#             err = F.interpolate(err, (H_quat, W_quat), mode='bilinear', align_corners=False)
#             ref = self.select_refinement_regions(err)
#             idx = torch.nonzero(ref.squeeze(1))
#             idx = idx[:, 0], idx[:, 1], idx[:, 2]
 
#             if idx[0].size(0) > 0:
#                 x = torch.cat([hid, pha, fgr], dim=1)
#                 x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
#                 x = self.crop_patch(x, idx, 2, 3 if self.kernel_size == 3 else 0)
 
#                 y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
#                 y = self.crop_patch(y, idx, 2, 3 if self.kernel_size == 3 else 0)
 
#                 x = self.conv1(torch.cat([x, y], dim=1))
#                 x = self.bn1(x)
#                 x = self.relu(x)
#                 x = self.conv2(x)
#                 x = self.bn2(x)
#                 x = self.relu(x)
 
#                 x = F.interpolate(x, 8 if self.kernel_size == 3 else 4, mode='nearest')
#                 y = self.crop_patch(src_bgr, idx, 4, 2 if self.kernel_size == 3 else 0)
 
#                 x = self.conv3(torch.cat([x, y], dim=1))
#                 x = self.bn3(x)
#                 x = self.relu(x)
#                 x = self.conv4(x)
 
#                 out = torch.cat([pha, fgr], dim=1)
#                 out = F.interpolate(out, (H_full, W_full), mode='bilinear', align_corners=False)
#                 out = self.replace_patch(out, x, idx)
#                 pha = out[:, :1]
#                 fgr = out[:, 1:]
#             else:
#                 pha = F.interpolate(pha, (H_full, W_full), mode='bilinear', align_corners=False)
#                 fgr = F.interpolate(fgr, (H_full, W_full), mode='bilinear', align_corners=False)
#         else:
#             x = torch.cat([hid, pha, fgr], dim=1)
#             x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
#             y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
#             if self.kernel_size == 3:
#                 x = F.pad(x, (3, 3, 3, 3))
#                 y = F.pad(y, (3, 3, 3, 3))
 
#             x = self.conv1(torch.cat([x, y], dim=1))
#             x = self.bn1(x)
#             x = self.relu(x)
#             x = self.conv2(x)
#             x = self.bn2(x)
#             x = self.relu(x)
 
#             if self.kernel_size == 3:
#                 x = F.interpolate(x, (H_full + 4, W_full + 4))
#                 y = F.pad(src_bgr, (2, 2, 2, 2))
#             else:
#                 x = F.interpolate(x, (H_full, W_full), mode='nearest')
#                 y = src_bgr
 
#             x = self.conv3(torch.cat([x, y], dim=1))
#             x = self.bn3(x)
#             x = self.relu(x)
#             x = self.conv4(x)
 
#             pha = x[:, :1]
#             fgr = x[:, 1:]
#             ref = torch.ones((src.size(0), 1, H_quat, W_quat), device=src.device, dtype=src.dtype)
 
#         return pha, fgr, ref
 
#     def select_refinement_regions(self, err: torch.Tensor):
#         """
#         Select refinement regions.
#         Input:
#             err: error map (B, 1, H, W)
#         Output:
#             ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
#         """
#         if self.mode == 'sampling':
#             # Sampling mode.
#             b, _, h, w = err.shape
#             err = err.view(b, -1)
#             idx = err.topk(self.sample_pixels // 16, dim=1, sorted=False).indices
#             ref = torch.zeros_like(err)
#             ref.scatter_(1, idx, 1.)
#             if self.prevent_oversampling:
#                 ref.mul_(err.gt(0).float())
#             ref = ref.view(b, 1, h, w)
#         else:
#             # Thresholding mode.
#             ref = err.gt(self.threshold).float()
#         return ref
 
#     def crop_patch(self,
#                    x: torch.Tensor,
#                    idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
#                    size: int,
#                    padding: int):
#         """
#         Crops selected patches from image given indices.
#         Inputs:
#             x: image (B, C, H, W).
#             idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
#             size: center size of the patch, also stride of the crop.
#             padding: expansion size of the patch.
#         Output:
#             patch: (P, C, h, w), where h = w = size + 2 * padding.
#         """
#         if padding != 0:
#             x = F.pad(x, (padding,) * 4)
 
#         if self.patch_crop_method == 'unfold':
#             # Use unfold. Best performance for PyTorch and TorchScript.
#             return x.permute(0, 2, 3, 1) \
#                 .unfold(1, size + 2 * padding, size) \
#                 .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]
#         elif self.patch_crop_method == 'roi_align':
#             # Use roi_align. Best compatibility for ONNX.
#             idx = idx[0].type_as(x), idx[1].type_as(x), idx[2].type_as(x)
#             b = idx[0]
#             x1 = idx[2] * size - 0.5
#             y1 = idx[1] * size - 0.5
#             x2 = idx[2] * size + size + 2 * padding - 0.5
#             y2 = idx[1] * size + size + 2 * padding - 0.5
#             boxes = torch.stack([b, x1, y1, x2, y2], dim=1)
#             return torchvision.ops.roi_align(x, boxes, size + 2 * padding, sampling_ratio=1)
#         else:
#             # Use gather. Crops out patches pixel by pixel.
#             idx_pix = self.compute_pixel_indices(x, idx, size, padding)
#             pat = torch.gather(x.view(-1), 0, idx_pix.view(-1))
#             pat = pat.view(-1, x.size(1), size + 2 * padding, size + 2 * padding)
#             return pat
 
#     def replace_patch(self,
#                       x: torch.Tensor,
#                       y: torch.Tensor,
#                       idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
#         """
#         Replaces patches back into image given index.
#         Inputs:
#             x: image (B, C, H, W)
#             y: patches (P, C, h, w)
#             idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.
#         Output:
#             image: (B, C, H, W), where patches at idx locations are replaced with y.
#         """
#         xB, xC, xH, xW = x.shape
#         yB, yC, yH, yW = y.shape
#         if self.patch_replace_method == 'scatter_nd':
#             # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
#             x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
#             x[idx[0], idx[1], idx[2]] = y
#             x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
#             return x
#         else:
#             # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
#             idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
#             return x.view(-1).scatter_(0, idx_pix.view(-1), y.view(-1)).view(x.shape)
 
#     def compute_pixel_indices(self,
#                               x: torch.Tensor,
#                               idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
#                               size: int,
#                               padding: int):
#         """
#         Compute selected pixel indices in the tensor.
#         Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
#         Input:
#             x: image: (B, C, H, W)
#             idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
#             size: center size of the patch, also stride of the crop.
#             padding: expansion size of the patch.
#         Output:
#             idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
#                  the element are indices pointing to the input x.view(-1).
#         """
#         B, C, H, W = x.shape
#         S, P = size, padding
#         O = S + 2 * P
#         b, y, x = idx
#         n = b.size(0)
#         c = torch.arange(C)
#         o = torch.arange(O)
#         idx_pat = (c * H * W).view(C, 1, 1).expand([C, O, O]) + (o * W).view(1, O, 1).expand([C, O, O]) + o.view(1, 1,
#                                                                                                                  O).expand(
#             [C, O, O])
#         idx_loc = b * W * H + y * W * S + x * S
#         idx_pix = idx_loc.view(-1, 1, 1, 1).expand([n, C, O, O]) + idx_pat.view(1, C, O, O).expand([n, C, O, O])
#         return idx_pix
 
 
# def load_matched_state_dict(model, state_dict, print_stats=True):
#     """
#     Only loads weights that matched in key and shape. Ignore other weights.
#     """
#     num_matched, num_total = 0, 0
#     curr_state_dict = model.state_dict()
#     for key in curr_state_dict.keys():
#         num_total += 1
#         if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
#             curr_state_dict[key] = state_dict[key]
#             num_matched += 1
#     model.load_state_dict(curr_state_dict)
#     if print_stats:
#         print(f'Loaded state_dict: {num_matched}/{num_total} matched')
 
 
# def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
 
 
# class ConvNormActivation(torch.nn.Sequential):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: int = 3,
#             stride: int = 1,
#             padding: Optional[int] = None,
#             groups: int = 1,
#             norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#             activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#             dilation: int = 1,
#             inplace: bool = True,
#     ) -> None:
#         if padding is None:
#             padding = (kernel_size - 1) // 2 * dilation
#         layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
#                                   dilation=dilation, groups=groups, bias=norm_layer is None)]
#         if norm_layer is not None:
#             layers.append(norm_layer(out_channels))
#         if activation_layer is not None:
#             layers.append(activation_layer(inplace=inplace))
#         super().__init__(*layers)
#         self.out_channels = out_channels
 
 
# class InvertedResidual(nn.Module):
#     def __init__(
#             self,
#             inp: int,
#             oup: int,
#             stride: int,
#             expand_ratio: int,
#             norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
 
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
 
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
 
#         layers: List[nn.Module] = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
#                                              activation_layer=nn.ReLU6))
#         layers.extend([
#             # dw
#             ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
#                                activation_layer=nn.ReLU6),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
#         self.out_channels = oup
#         self._is_cn = stride > 1
 
#     def forward(self, x: Tensor) -> Tensor:
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
 
 
# class MobileNetV2(nn.Module):
#     def __init__(
#             self,
#             num_classes: int = 1000,
#             width_mult: float = 1.0,
#             inverted_residual_setting: Optional[List[List[int]]] = None,
#             round_nearest: int = 8,
#             block: Optional[Callable[..., nn.Module]] = None,
#             norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         """
#         MobileNet V2 main class
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#             block: Module specifying inverted residual building block for mobilenet
#             norm_layer: Module specifying the normalization layer to use
#         """
#         super(MobileNetV2, self).__init__()
 
#         if block is None:
#             block = InvertedResidual
 
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
 
#         input_channel = 32
#         last_channel = 1280
 
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]
 
#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))
 
#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
#                                                         activation_layer=nn.ReLU6)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
#                                            activation_layer=nn.ReLU6))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)
 
#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
 
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
 
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # This exists since TorchScript doesn't support inheritance, so the superclass method
#         # (this one) needs to have a name other than `forward` that can be accessed in a subclass
#         x = self.features(x)
#         # Cannot use "squeeze" as batch-size can be 1
#         x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
 
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)
 
 
# class MobileNetV2Encoder(MobileNetV2):
#     """
#     MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
#     use dilation on the last block to maintain output stride 16, and deleted the
#     classifier block that was originally used for classification. The forward method
#     additionally returns the feature maps at all resolutions for decoder's use.
#     """
 
#     def __init__(self, in_channels, norm_layer=None):
#         super().__init__()
 
#         # Replace first conv layer if in_channels doesn't match.
#         if in_channels != 3:
#             self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
 
#         # Remove last block
#         self.features = self.features[:-1]
 
#         # Change to use dilation to maintain output stride = 16
#         self.features[14].conv[1][0].stride = (1, 1)
#         for feature in self.features[15:]:
#             feature.conv[1][0].dilation = (2, 2)
#             feature.conv[1][0].padding = (2, 2)
 
#         # Delete classifier
#         del self.classifier
 
#     def forward(self, x):
#         x0 = x  # 1/1
#         x = self.features[0](x)
#         x = self.features[1](x)
#         x1 = x  # 1/2
#         x = self.features[2](x)
#         x = self.features[3](x)
#         x2 = x  # 1/4
#         x = self.features[4](x)
#         x = self.features[5](x)
#         x = self.features[6](x)
#         x3 = x  # 1/8
#         x = self.features[7](x)
#         x = self.features[8](x)
#         x = self.features[9](x)
#         x = self.features[10](x)
#         x = self.features[11](x)
#         x = self.features[12](x)
#         x = self.features[13](x)
#         x = self.features[14](x)
#         x = self.features[15](x)
#         x = self.features[16](x)
#         x = self.features[17](x)
#         x4 = x  # 1/16
#         return x4, x3, x2, x1, x0
 
 
# class Decoder(nn.Module):
 
#     def __init__(self, channels, feature_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channels[1])
#         self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels[2])
#         self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(channels[3])
#         self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
#         self.relu = nn.ReLU(True)
 
#     def forward(self, x4, x3, x2, x1, x0):
#         x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, x0], dim=1)
#         x = self.conv4(x)
#         return x
 
 
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
 
 
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
 
 
# class ASPP(nn.Module):
#     def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
#         super(ASPP, self).__init__()
#         modules = []
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
 
#         rates = tuple(atrous_rates)
#         for rate in rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
 
#         modules.append(ASPPPooling(in_channels, out_channels))
 
#         self.convs = nn.ModuleList(modules)
 
#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _res = []
#         for conv in self.convs:
#             _res.append(conv(x))
#         res = torch.cat(_res, dim=1)
#         return self.project(res)
 
 
# class ResNetEncoder(ResNet):
#     layers = {
#         'resnet50': [3, 4, 6, 3],
#         'resnet101': [3, 4, 23, 3],
#     }
 
#     def __init__(self, in_channels, variant='resnet101', norm_layer=None):
#         super().__init__(
#             block=Bottleneck,
#             layers=self.layers[variant],
#             replace_stride_with_dilation=[False, False, True],
#             norm_layer=norm_layer)
 
#         # Replace first conv layer if in_channels doesn't match.
#         if in_channels != 3:
#             self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
 
#         # Delete fully-connected layer
#         del self.avgpool
#         del self.fc
 
#     def forward(self, x):
#         x0 = x  # 1/1
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x1 = x  # 1/2
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x2 = x  # 1/4
#         x = self.layer2(x)
#         x3 = x  # 1/8
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x4 = x  # 1/16
#         return x4, x3, x2, x1, x0
 
 
# class Base(nn.Module):
#     """
#     A generic implementation of the base encoder-decoder network inspired by DeepLab.
#     Accepts arbitrary channels for input and output.
#     """
 
#     def __init__(self, backbone: str, in_channels: int, out_channels: int):
#         super().__init__()
#         assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
#         if backbone in ['resnet50', 'resnet101']:
#             self.backbone = ResNetEncoder(in_channels, variant=backbone)
#             self.aspp = ASPP(2048, [3, 6, 9])
#             self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
#         else:
#             self.backbone = MobileNetV2Encoder(in_channels)
#             self.aspp = ASPP(320, [3, 6, 9])
#             self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])
 
#     def forward(self, x):
#         x, *shortcuts = self.backbone(x)
#         x = self.aspp(x)
#         x = self.decoder(x, *shortcuts)
#         return x
 
#     def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
#         # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
#         # This method converts and loads their pretrained state_dict to match with our model structure.
#         # This method is not needed if you are not planning to train from deeplab weights.
#         # Use load_state_dict() for normal weight loading.
 
#         # Convert state_dict naming for aspp module
#         state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}
 
#         if isinstance(self.backbone, ResNetEncoder):
#             # ResNet backbone does not need change.
#             load_matched_state_dict(self, state_dict, print_stats)
#         else:
#             # Change MobileNetV2 backbone to state_dict format, then change back after loading.
#             backbone_features = self.backbone.features
#             self.backbone.low_level_features = backbone_features[:4]
#             self.backbone.high_level_features = backbone_features[4:]
#             del self.backbone.features
#             load_matched_state_dict(self, state_dict, print_stats)
#             self.backbone.features = backbone_features
#             del self.backbone.low_level_features
#             del self.backbone.high_level_features
 
 
# class MattingBase(Base):
 
#     def __init__(self, backbone: str):
#         super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))
 
#     def forward(self, src, bgr):
#         x = torch.cat([src, bgr], dim=1)
#         x, *shortcuts = self.backbone(x)
#         x = self.aspp(x)
#         x = self.decoder(x, *shortcuts)
#         pha = x[:, 0:1].clamp_(0., 1.)
#         fgr = x[:, 1:4].add(src).clamp_(0., 1.)
#         err = x[:, 4:5].clamp_(0., 1.)
#         hid = x[:, 5:].relu_()
#         return pha, fgr, err, hid
 
 
# class MattingRefine(MattingBase):
 
#     def __init__(self,
#                  backbone: str,
#                  backbone_scale: float = 1 / 4,
#                  refine_mode: str = 'sampling',
#                  refine_sample_pixels: int = 80_000,
#                  refine_threshold: float = 0.1,
#                  refine_kernel_size: int = 3,
#                  refine_prevent_oversampling: bool = True,
#                  refine_patch_crop_method: str = 'unfold',
#                  refine_patch_replace_method: str = 'scatter_nd'):
#         assert backbone_scale <= 1 / 2, 'backbone_scale should not be greater than 1/2'
#         super().__init__(backbone)
#         self.backbone_scale = backbone_scale
#         self.refiner = Refiner(refine_mode,
#                                refine_sample_pixels,
#                                refine_threshold,
#                                refine_kernel_size,
#                                refine_prevent_oversampling,
#                                refine_patch_crop_method,
#                                refine_patch_replace_method)
 
#     def forward(self, src, bgr):
#         assert src.size() == bgr.size(), 'src and bgr must have the same shape'
#         assert src.size(2) // 4 * 4 == src.size(2) and src.size(3) // 4 * 4 == src.size(3), \
#             'src and bgr must have width and height that are divisible by 4'
 
#         # Downsample src and bgr for backbone
#         src_sm = F.interpolate(src,
#                                scale_factor=self.backbone_scale,
#                                mode='bilinear',
#                                align_corners=False,
#                                recompute_scale_factor=True)
#         bgr_sm = F.interpolate(bgr,
#                                scale_factor=self.backbone_scale,
#                                mode='bilinear',
#                                align_corners=False,
#                                recompute_scale_factor=True)
 
#         # Base
#         x = torch.cat([src_sm, bgr_sm], dim=1)
#         x, *shortcuts = self.backbone(x)
#         x = self.aspp(x)
#         x = self.decoder(x, *shortcuts)
#         pha_sm = x[:, 0:1].clamp_(0., 1.)
#         fgr_sm = x[:, 1:4]
#         err_sm = x[:, 4:5].clamp_(0., 1.)
#         hid_sm = x[:, 5:].relu_()
 
#         # Refiner
#         pha, fgr, ref_sm = self.refiner(src, bgr, pha_sm, fgr_sm, err_sm, hid_sm)
 
#         # Clamp outputs
#         pha = pha.clamp_(0., 1.)
#         fgr = fgr.add_(src).clamp_(0., 1.)
#         fgr_sm = src_sm.add_(fgr_sm).clamp_(0., 1.)
 
#         return pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm
 
 
# class ImagesDataset(Dataset):
#     def __init__(self, root, mode='RGB', transforms=None):
#         self.transforms = transforms
#         self.mode = mode
#         self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
#                                  *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])
 
#     def __len__(self):
#         return len(self.filenames)
 
#     def __getitem__(self, idx):
#         with Image.open(self.filenames[idx]) as img:
#             img = img.convert(self.mode)
#         if self.transforms:
#             img = self.transforms(img)
 
#         return img
 
 
# class NewImagesDataset(Dataset):
#     def __init__(self, root, mode='RGB', transforms=None):
#         self.transforms = transforms
#         self.mode = mode
#         self.filenames = [root]
#         print(self.filenames)
 
#     def __len__(self):
#         return len(self.filenames)
 
#     def __getitem__(self, idx):
#         with Image.open(self.filenames[idx]) as img:
#             img = img.convert(self.mode)
 
#         if self.transforms:
#             img = self.transforms(img)
 
#         return img
 
 
# class ZipDataset(Dataset):
#     def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False):
#         self.datasets = datasets
#         self.transforms = transforms
 
#         if assert_equal_length:
#             for i in range(1, len(datasets)):
#                 assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
 
#     def __len__(self):
#         return max(len(d) for d in self.datasets)
 
#     def __getitem__(self, idx):
#         x = tuple(d[idx % len(d)] for d in self.datasets)
#         print(x)
#         if self.transforms:
#             x = self.transforms(*x)
#         return x
 
 
# class PairCompose(T.Compose):
#     def __call__(self, *x):
#         for transform in self.transforms:
#             x = transform(*x)
#         return x
 
 
# class PairApply:
#     def __init__(self, transforms):
#         self.transforms = transforms
 
#     def __call__(self, *x):
#         return [self.transforms(xi) for xi in x]
 
 
# # --------------- Arguments ---------------
 
# parser = argparse.ArgumentParser(description='hy-replace-background')
 
# parser.add_argument('--model-type', type=str, required=False, choices=['mattingbase', 'mattingrefine'],
#                     default='mattingrefine')
# parser.add_argument('--model-backbone', type=str, required=False, choices=['resnet101', 'resnet50', 'mobilenetv2'],
#                     default='resnet50')
# parser.add_argument('--model-backbone-scale', type=float, default=0.25)
# parser.add_argument('--model-checkpoint', type=str, required=False, default='model/pytorch_resnet50.pth')
# parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
# parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
# parser.add_argument('--model-refine-threshold', type=float, default=0.7)
# parser.add_argument('--model-refine-kernel-size', type=int, default=3)
 
# parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
# parser.add_argument('--num-workers', type=int, default=0,
#                     help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
# parser.add_argument('--preprocess-alignment', action='store_true')
 
# parser.add_argument('--output-dir', type=str, required=False, default='content/output')
# parser.add_argument('--output-types', type=str, required=False, nargs='+',
#                     choices=['com', 'pha', 'fgr', 'err', 'ref', 'new'],
#                     default=['new'])
# parser.add_argument('-y', action='store_true')
 
 
# def handle(image_path: str, bgr_path: str, new_bg: str):
#     parser.add_argument('--images-src', type=str, required=False, default=image_path)
#     parser.add_argument('--images-bgr', type=str, required=False, default=bgr_path)
#     args = parser.parse_args()
 
#     assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
#         'Only mattingbase and mattingrefine support err output'
#     assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
#         'Only mattingrefine support ref output'
 
#     # --------------- Main ---------------
 
#     device = torch.device(args.device)
 
#     # Load model
#     if args.model_type == 'mattingbase':
#         model = MattingBase(args.model_backbone)
#     if args.model_type == 'mattingrefine':
#         model = MattingRefine(
#             args.model_backbone,
#             args.model_backbone_scale,
#             args.model_refine_mode,
#             args.model_refine_sample_pixels,
#             args.model_refine_threshold,
#             args.model_refine_kernel_size)
 
#     model = model.to(device).eval()
#     model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)
 
#     # Load images
#     dataset = ZipDataset([
#         NewImagesDataset(args.images_src),
#         NewImagesDataset(args.images_bgr),
#     ], assert_equal_length=True, transforms=PairCompose([
#         HomographicAlignment() if args.preprocess_alignment else PairApply(nn.Identity()),
#         PairApply(T.ToTensor())
#     ]))
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
 
#     # # Create output directory
#     # if os.path.exists(args.output_dir):
#     #     if args.y or input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
#     #         shutil.rmtree(args.output_dir)
#     #     else:
#     #         exit()
 
#     for output_type in args.output_types:
#         if os.path.exists(os.path.join(args.output_dir, output_type)) is False:
#             os.makedirs(os.path.join(args.output_dir, output_type))
 
#     # Worker function
#     def writer(img, path):
#         img = to_pil_image(img[0].cpu())
#         img.save(path)
 
#     # Worker function
#     def writer_hy(img, new_bg, path):
#         img = to_pil_image(img[0].cpu())
#         img_size = img.size
#         new_bg_img = Image.open(new_bg).convert('RGBA')
#         new_bg_img.resize(img_size, Image.ANTIALIAS)
#         out = Image.alpha_composite(new_bg_img, img)
#         out.save(path)
 
#     result_file_name = str(uuid.uuid4())
 
#     # Conversion loop
#     with torch.no_grad():
#         for i, (src, bgr) in enumerate(tqdm(dataloader)):
#             src = src.to(device, non_blocking=True)
#             bgr = bgr.to(device, non_blocking=True)
 
#             if args.model_type == 'mattingbase':
#                 pha, fgr, err, _ = model(src, bgr)
#             elif args.model_type == 'mattingrefine':
#                 pha, fgr, _, _, err, ref = model(src, bgr)
 
#             pathname = dataset.datasets[0].filenames[i]
#             pathname = os.path.relpath(pathname, args.images_src)
#             pathname = os.path.splitext(pathname)[0]
 
#             if 'new' in args.output_types:
#                 new = torch.cat([fgr * pha.ne(0), pha], dim=1)
#                 Thread(target=writer_hy,
#                        args=(new, new_bg, os.path.join(args.output_dir, 'new', result_file_name + '.png'))).start()
#             if 'com' in args.output_types:
#                 com = torch.cat([fgr * pha.ne(0), pha], dim=1)
#                 Thread(target=writer, args=(com, os.path.join(args.output_dir, 'com', pathname + '.png'))).start()
#             if 'pha' in args.output_types:
#                 Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha', pathname + '.jpg'))).start()
#             if 'fgr' in args.output_types:
#                 Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr', pathname + '.jpg'))).start()
#             if 'err' in args.output_types:
#                 err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
#                 Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err', pathname + '.jpg'))).start()
#             if 'ref' in args.output_types:
#                 ref = F.interpolate(ref, src.shape[2:], mode='nearest')
#                 Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname + '.jpg'))).start()
 
#     return os.path.join(args.output_dir, 'new', result_file_name + '.png')
 
 
# if __name__ == '__main__':
#     handle("data/img2.png", "data/bg.png", "data/newbg.jpg")







#--------------------------------------------zed-----------------------------------------------------------#
# import os
# import sys
# import time
# import logging
# import random
# from random import randint
# import math
# import statistics
# import getopt
# from ctypes import *
# import numpy as np
# import cv2
# import pyzed.sl as sl

# # Get the top-level logger object
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# def sample(probs):
#     s = sum(probs)
#     probs = [a/s for a in probs]
#     r = random.uniform(0, 1)
#     for i in range(len(probs)):
#         r = r - probs[i]
#         if r <= 0:
#             return i
#     return len(probs)-1


# def c_array(ctype, values):
#     arr = (ctype*len(values))()
#     arr[:] = values
#     return arr


# class BOX(Structure):
#     _fields_ = [("x", c_float),
#                 ("y", c_float),
#                 ("w", c_float),
#                 ("h", c_float)]


# class DETECTION(Structure):
#     _fields_ = [("bbox", BOX),
#                 ("classes", c_int),
#                 ("prob", POINTER(c_float)),
#                 ("mask", POINTER(c_float)),
#                 ("objectness", c_float),
#                 ("sort_class", c_int),
#                 ("uc", POINTER(c_float)),
#                 ("points", c_int),
#                 ("embeddings", POINTER(c_float)),
#                 ("embedding_size", c_int),
#                 ("sim", c_float),
#                 ("track_id", c_int)]


# class IMAGE(Structure):
#     _fields_ = [("w", c_int),
#                 ("h", c_int),
#                 ("c", c_int),
#                 ("data", POINTER(c_float))]


# class METADATA(Structure):
#     _fields_ = [("classes", c_int),
#                 ("names", POINTER(c_char_p))]


# #lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# #lib = CDLL("darknet.so", RTLD_GLOBAL)
# hasGPU = True
# if os.name == "nt":
#     cwd = os.path.dirname(__file__)
#     os.environ['PATH'] = cwd + ';' + os.environ['PATH']
#     winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
#     winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
#     envKeys = list()
#     for k, v in os.environ.items():
#         envKeys.append(k)
#     try:
#         try:
#             tmp = os.environ["FORCE_CPU"].lower()
#             if tmp in ["1", "true", "yes", "on"]:
#                 raise ValueError("ForceCPU")
#             else:
#                 log.info("Flag value '"+tmp+"' not forcing CPU mode")
#         except KeyError:
#             # We never set the flag
#             if 'CUDA_VISIBLE_DEVICES' in envKeys:
#                 if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
#                     raise ValueError("ForceCPU")
#             try:
#                 global DARKNET_FORCE_CPU
#                 if DARKNET_FORCE_CPU:
#                     raise ValueError("ForceCPU")
#             except NameError:
#                 pass
#             # log.info(os.environ.keys())
#             # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
#         if not os.path.exists(winGPUdll):
#             raise ValueError("NoDLL")
#         lib = CDLL(winGPUdll, RTLD_GLOBAL)
#     except (KeyError, ValueError):
#         hasGPU = False
#         if os.path.exists(winNoGPUdll):
#             lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
#             log.warning("Notice: CPU-only mode")
#         else:
#             # Try the other way, in case no_gpu was
#             # compile but not renamed
#             lib = CDLL(winGPUdll, RTLD_GLOBAL)
#             log.warning("Environment variables indicated a CPU run, but we didn't find `" +
#                         winNoGPUdll+"`. Trying a GPU run anyway.")
# else:
#     # lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)
#     lib = CDLL("./libdarknet/libdarknet.so", RTLD_GLOBAL)
# lib.network_width.argtypes = [c_void_p]
# lib.network_width.restype = c_int
# lib.network_height.argtypes = [c_void_p]
# lib.network_height.restype = c_int

# predict = lib.network_predict
# predict.argtypes = [c_void_p, POINTER(c_float)]
# predict.restype = POINTER(c_float)

# if hasGPU:
#     set_gpu = lib.cuda_set_device
#     set_gpu.argtypes = [c_int]

# make_image = lib.make_image
# make_image.argtypes = [c_int, c_int, c_int]
# make_image.restype = IMAGE

# get_network_boxes = lib.get_network_boxes
# get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
#     c_int), c_int, POINTER(c_int), c_int]
# get_network_boxes.restype = POINTER(DETECTION)

# make_network_boxes = lib.make_network_boxes
# make_network_boxes.argtypes = [c_void_p]
# make_network_boxes.restype = POINTER(DETECTION)

# free_detections = lib.free_detections
# free_detections.argtypes = [POINTER(DETECTION), c_int]

# free_ptrs = lib.free_ptrs
# free_ptrs.argtypes = [POINTER(c_void_p), c_int]

# network_predict = lib.network_predict
# network_predict.argtypes = [c_void_p, POINTER(c_float)]

# reset_rnn = lib.reset_rnn
# reset_rnn.argtypes = [c_void_p]

# load_net = lib.load_network
# load_net.argtypes = [c_char_p, c_char_p, c_int]
# load_net.restype = c_void_p

# load_net_custom = lib.load_network_custom
# load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
# load_net_custom.restype = c_void_p

# do_nms_obj = lib.do_nms_obj
# do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# do_nms_sort = lib.do_nms_sort
# do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# free_image = lib.free_image
# free_image.argtypes = [IMAGE]

# letterbox_image = lib.letterbox_image
# letterbox_image.argtypes = [IMAGE, c_int, c_int]
# letterbox_image.restype = IMAGE

# load_meta = lib.get_metadata
# lib.get_metadata.argtypes = [c_char_p]
# lib.get_metadata.restype = METADATA

# load_image = lib.load_image_color
# load_image.argtypes = [c_char_p, c_int, c_int]
# load_image.restype = IMAGE

# rgbgr_image = lib.rgbgr_image
# rgbgr_image.argtypes = [IMAGE]

# predict_image = lib.network_predict_image
# predict_image.argtypes = [c_void_p, IMAGE]
# predict_image.restype = POINTER(c_float)


# def array_to_image(arr):
#     import numpy as np
#     # need to return old values to avoid python freeing memory
#     arr = arr.transpose(2, 0, 1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
#     data = arr.ctypes.data_as(POINTER(c_float))
#     im = IMAGE(w, h, c, data)
#     return im, arr


# def classify(net, meta, im):
#     out = predict_image(net, im)
#     res = []
#     for i in range(meta.classes):
#         if altNames is None:
#             name_tag = meta.names[i]
#         else:
#             name_tag = altNames[i]
#         res.append((name_tag, out[i]))
#     res = sorted(res, key=lambda x: -x[1])
#     return res


# def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
#     """
#     Performs the detection
#     """
#     custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     custom_image = cv2.resize(custom_image, (lib.network_width(net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
#     im, arr = array_to_image(custom_image)
#     num = c_int(0)
#     pnum = pointer(num)
#     predict_image(net, im)
#     dets = get_network_boxes(net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
#     num = pnum[0]
#     if nms:
#         do_nms_sort(dets, num, meta.classes, nms)
#     res = []
#     if debug:
#         log.debug("about to range")
#     for j in range(num):
#         for i in range(meta.classes):
#             if dets[j].prob[i] > 0:
#                 b = dets[j].bbox
#                 if altNames is None:
#                     name_tag = meta.names[i]
#                 else:
#                     name_tag = altNames[i]
#                 res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
#     res = sorted(res, key=lambda x: -x[1])
#     free_detections(dets, num)
#     return res


# netMain = None
# metaMain = None
# altNames = None


# def get_object_depth(depth, bounds):
#     '''
#     Calculates the median x, y, z position of top slice(area_div) of point cloud
#     in camera frame.
#     Arguments:
#         depth: Point cloud data of whole frame.
#         bounds: Bounding box for object in pixels.
#             bounds[0]: x-center
#             bounds[1]: y-center
#             bounds[2]: width of bounding box.
#             bounds[3]: height of bounding box.

#     Return:
#         x, y, z: Location of object in meters.
#     '''
#     area_div = 2

#     x_vect = []
#     y_vect = []
#     z_vect = []

#     for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
#         for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
#             z = depth[i, j, 2]
#             if not np.isnan(z) and not np.isinf(z):
#                 x_vect.append(depth[i, j, 0])
#                 y_vect.append(depth[i, j, 1])
#                 z_vect.append(z)
#     try:
#         x_median = statistics.median(x_vect)
#         y_median = statistics.median(y_vect)
#         z_median = statistics.median(z_vect)
#     except Exception:
#         x_median = -1
#         y_median = -1
#         z_median = -1
#         pass

#     return x_median, y_median, z_median


# def generate_color(meta_path):
#     '''
#     Generate random colors for the number of classes mentioned in data file.
#     Arguments:
#     meta_path: Path to .data file.

#     Return:
#     color_array: RGB color codes for each class.
#     '''
#     random.seed(42)
#     with open(meta_path, 'r') as f:
#         content = f.readlines()
#     class_num = int(content[0].split("=")[1])
#     color_array = []
#     for x in range(0, class_num):
#         color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     return color_array


# def main(argv):
#     print(argv)

#     thresh = 0.7
#     # thresh = 0.7
#     darknet_path="./libdarknet/"
#     # darknet_path="../libdarknet/"
#     config_path = darknet_path + "cfg/yolov4-obj.cfg"
#     weight_path = "./yolo_data/yolov4-obj_best.weights"
#     meta_path = "./yolo_data/obj.data"
#     svo_path = None
#     zed_id = 0

#     # help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
#     # try:
#     #     opts, args = getopt.getopt(argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
#     #     print(opts,args)
#     # except getopt.GetoptError:
#     #     log.exception(help_str)
#     #     sys.exit(2)
#     # for opt, arg in opts:
#     #     # if opt == '-h':
#     #     #     log.info(help_str)
#     #     #     sys.exit()
#     #     if opt in ("-c", "--config"):
#     #         config_path = arg
#     #         print(config_path)
#     #     elif opt in ("-w", "--weight"):
#     #         weight_path = arg
#     #     elif opt in ("-m", "--meta"):
#     #         meta_path = arg
#     #     elif opt in ("-t", "--threshold"):
#     #         thresh = float(arg)
#     #     elif opt in ("-s", "--svo_file"):
#     #         svo_path = arg
#     #     elif opt in ("-z", "--zed_id"):
#     #         zed_id = int(arg)

#     cam = sl.Camera()
#     input_type = sl.InputType()
#     # if svo_path is not None:
#     #     log.info("SVO file : " + svo_path)
#     #     input_type.set_from_svo_file(svo_path)
#     # else:
#     #     # Launch camera by id

#     input_type.set_from_camera_id(zed_id)

#     init = sl.InitParameters(input_t=input_type)
#     init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
#     init.camera_fps = 10
#     init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
#     init.coordinate_units = sl.UNIT.METER

    
#     if not cam.is_opened():
#         log.info("Opening ZED Camera...")
#     status = cam.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         log.error(repr(status))
#         exit()

#     runtime = sl.RuntimeParameters()
#     runtime.sensing_mode = sl.SENSING_MODE.STANDARD

#     #set image size
#     image_size = cam.get_camera_information().camera_resolution
#     image_size.width = 1280
#     image_size.height = 720
    
#     mat = sl.Mat()
#     point_cloud_mat = sl.Mat()

#     image_zed_left = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
#     image_zed_right = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
#     depth_image_zed = sl.Mat(image_size.width,image_size.height, sl.MAT_TYPE.U8_C4)

#     # Import the global variables. This lets us instance Darknet once,
#     # then just call performDetect() again without instancing again
#     global metaMain, netMain, altNames  # pylint: disable=W0603
#     assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
#     if not os.path.exists(config_path):
#         raise ValueError("Invalid config path `" +
#                          os.path.abspath(config_path)+"`")
#     if not os.path.exists(weight_path):
#         raise ValueError("Invalid weight path `" +
#                          os.path.abspath(weight_path)+"`")
#     if not os.path.exists(meta_path):
#         raise ValueError("Invalid data file path `" +
#                          os.path.abspath(meta_path)+"`")
#     if netMain is None:
#         netMain = load_net_custom(config_path.encode(
#             "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
#     if metaMain is None:
#         metaMain = load_meta(meta_path.encode("ascii"))
#     if altNames is None:
#         # In thon 3, the metafile default access craps out on Windows (but not Linux)
#         # Read the names file and create a list to feed to detect
#         try:
#             with open(meta_path) as meta_fh:
#                 meta_contents = meta_fh.read()
#                 import re
#                 match = re.search("names *= *(.*)$", meta_contents,
#                                   re.IGNORECASE | re.MULTILINE)
#                 if match:
#                     result = match.group(1)
#                 else:
#                     result = None
#                 try:
#                     if os.path.exists(result):
#                         with open(result) as names_fh:
#                             names_list = names_fh.read().strip().split("\n")
#                             altNames = [x.strip() for x in names_list]
#                 except TypeError:
#                     pass
#         except Exception:
#             pass

#     color_array = generate_color(meta_path)

#     log.info("Running...")

#     key = ''
#     while key != 113:  # for 'q' key
#         start_time = time.time() # start time of the loop
#         err = cam.grab(runtime)
#         if err == sl.ERROR_CODE.SUCCESS:

#             cam.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
#             cam.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
#             cam.retrieve_measure(depth_image_zed, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

#             image_left = image_zed_left.get_data()
#             # image_right = image_zed_left.get_data()
#             depth_image = depth_image_zed.get_data()
            
#             # cam.retrieve_image(mat, sl.VIEW.LEFT)
#             # image = mat.get_data()

#             # cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
#             # depth = point_cloud_mat.get_data()

#             # Do the detection
#             detections = detect(netMain, metaMain, image_left, thresh)
#             # print(detections)

#             # log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
#             for detection in detections:
#                 label = detection[0]
#                 confidence = detection[1]
#                 pstring = label+": "+str(np.rint(100 * confidence))+"%"
#                 log.info(pstring)
#                 bounds = detection[2]
#                 y_extent = int(bounds[3])
#                 x_extent = int(bounds[2])
#                 # Coordinates are around the center
#                 x_coord = int(bounds[0] - bounds[2]/2)
#                 y_coord = int(bounds[1] - bounds[3]/2)
#                 #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
#                 thickness = 1
#                 x, y, z = get_object_depth(depth_image, bounds)
#                 distance = math.sqrt(x * x + y * y + z * z)
#                 distance = "{:.2f}".format(distance)
#                 cv2.rectangle(image_left, (x_coord - thickness, y_coord - thickness),
#                               (x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
#                               color_array[detection[3]], -1)
#                 cv2.putText(image_left, label + " " +  (str(distance) + " m"),
#                             (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 cv2.rectangle(image_left, (x_coord - thickness, y_coord - thickness),
#                               (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
#                               color_array[detection[3]], int(thickness*2))

#             cv2.imshow("ZED_left", image_left)
#             cv2.imshow("ZED-depth", depth_image)
#             key = cv2.waitKey(5)
#             log.info("FPS: {}".format(1.0 / (time.time() - start_time)))
#         else:
#             key = cv2.waitKey(5)
#     cv2.destroyAllWindows()

#     cam.close()
#     log.info("\nFINISH")


# if __name__ == "__main__":
#     main(sys.argv[1:])






#備份
# def handle(image_path, bgr_path):

#     # set device
#     device = torch.device(args.device)
#     # Load model
#     if args.model_type == 'mattingbase':
#         model = MattingBase(args.model_backbone)
#     if args.model_type == 'mattingrefine':
#         model = MattingRefine(
#             args.model_backbone,
#             args.model_backbone_scale,
#             args.model_refine_mode,
#             args.model_refine_sample_pixels,
#             args.model_refine_threshold,
#             args.model_refine_kernel_size)

#     model = model.to(device).eval()
#     model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

#     #load image
#     imglist = sorted(os.listdir(image_path))
#     bgrlist = sorted(os.listdir(bgr_path))
#     count_img = len(imglist)
    

#     for i in range(count_img):
#         filester = imglist[i].split(".")[0]
#         bground_path = bgr_path + filester +".jpg"
#         img_path = image_path + filester +".jpg"
#         print(img_path)
        
#         # print(bground_path)

#         assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
#             'Only mattingbase and mattingrefine support err output'
#         assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
#             'Only mattingrefine support ref output'

#         # parser.add_argument('--images-bgr', type=str, required=False, default=img_path)
#         # parser.add_argument('--images-bgr', type=str, required=False, default=bground_path)
    
#         # args = parser.parse_args()
#         # print(args.img_src)

#         # --------------- Main ---------------  
#         # set imgfile
#         dataset = ZipDataset([
#             NewImagesDataset(img_path),
#             NewImagesDataset(bground_path),
#         ], assert_equal_length=True, transforms=PairCompose([
#             HomographicAlignment() if args.preprocess_alignment else PairApply(nn.Identity()),
#             PairApply(T.ToTensor())
#         ]))
        
#         dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
        
#         # Worker function
#         def writer(img, path):
#             img = to_pil_image(img[0].cpu())
#             #print(path)
#             img.save(path)
     
#         # print(dataloader)
        
#         with torch.no_grad():
#             for (src, bgr) in dataloader:
#                 src = src.to(device, non_blocking=True)
#                 bgr = bgr.to(device, non_blocking=True)

#                 if args.model_type == 'mattingbase':
#                     pha, fgr, err, _ = model(src, bgr)
#                 elif args.model_type == 'mattingrefine':
#                     pha, fgr, _, _, err, ref = model(src, bgr)
                
#                 com = torch.cat([fgr * pha.ne(0), pha], dim=1)
#                 # a = tensor_to_np(pha)
#                 # cv2.imshow("1111",a)

#                 # for i in str(a):
#                 #             
#                 #             cv2.imwrite("/home/user/shape_detection/circle/"+"long_"+str(a)+'.jpg',img)
            

#                 # print(dataset.datasets[0])
#                 # pathname = dataset.datasets[0].filenames[i]
#                 # print(pathname)

#                 # pathname1 = os.path.relpath(pathname, img_path)
#                 # # print(pathname1)

#                 # pathname2 = os.path.splitext(pathname)[0]
#                 # # print(pathname2)
            
#                 # if 'new' in args.output_types:
#                 #     new = torch.cat([fgr * pha.ne(0), pha], dim=1)
#                 #     Thread(target=writer,args=(new, new_bg, os.path.join(args.output_dir, 'new', result_file_name + '.png'))).start()

#                 # if 'com' in args.output_types:
#                 #     com = torch.cat([fgr * pha.ne(0), pha], dim=1)
#                 #     Thread(target=writer, args=(com, os.path.join(args.output_dir, filester+'_com' + '.png'))).start()

#                 # if 'pha' in args.output_types:
#                 #     Thread(target=writer, args=(pha, os.path.join(args.output_dir, filester +'_pha'  + '.jpg'))).start()

#                 # if 'fgr' in args.output_types:
#                 #     Thread(target=writer, args=(fgr, os.path.join(args.output_dir, filester +'_fgr' + '.jpg'))).start()

#                 # if 'err' in args.output_types:
#                 #     err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
#                 #     Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err',  filester_img + '_err'+ '.jpg'))).start()

#                 # if 'ref' in args.output_types:
#                 #     ref = F.interpolate(ref, src.shape[2:], mode='nearest')
#                 #     Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname +filester_img + '.jpg'))).start()
#     return com,fgr,pha,img_path