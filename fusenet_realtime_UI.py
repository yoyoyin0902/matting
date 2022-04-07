# Imports libraries
import os, cv2, torch, sys, time
import depthai as dai
import numpy as np
import torchvision.transforms as T
from PIL import Image
# from options.demo_options import DemoOptions
# from models import create_model_without_dataset
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import qtmodern.styles
import qtmodern.windows

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------- #

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0



print("[INFO]...READY...")

# ---------------------------------------------------------- #
##############################################################
##################### Initializing Camera ####################
##############################################################
class CameraWorker(QThread):
    frame = pyqtSignal(np.ndarray)
    frame_depth8 = pyqtSignal(np.ndarray)
    frame_depth16 = pyqtSignal(np.ndarray)

    # @pyqtSlot
    def __init__(self):
        super(CameraWorker, self).__init__()

    def run(self):
        downscaleColor = True
        # Create pipeline
        pipeline = dai.Pipeline()
        queueNames = []
        # Define sources and outputs
        camRgb = pipeline.createColorCamera()
        left = pipeline.createMonoCamera()
        right = pipeline.createMonoCamera()
        stereo = pipeline.createStereoDepth()

        rgbOut = pipeline.createXLinkOut()
        depthOut = pipeline.createXLinkOut()

        rgbOut.setStreamName("rgb")
        queueNames.append("rgb")
        depthOut.setStreamName("depth")
        queueNames.append("depth")

        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setFps(10)
        if downscaleColor: camRgb.setIspScale(2, 3)

        camRgb.initialControl.setManualFocus(130)

        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        left.setFps(5)
        right.setFps(5)

        stereo.setConfidenceThreshold(245)
        stereo.setRectifyEdgeFillColor(0)
        # LR-check is required for depth alignment
        stereo.setLeftRightCheck(True)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        camRgb.isp.link(rgbOut.input)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.depth.link(depthOut.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            device.getOutputQueue(name="rgb",   maxSize=4, blocking=False)
            device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            self.frameRgb = None
            self.frameDepth16 = None
            self.frameDepth8 = None
            while True:
                latestPacket = {}
                latestPacket["rgb"] = None
                latestPacket["depth"] = None

                queueEvents = device.getQueueEvents(("rgb", "depth"))
                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        latestPacket[queueName] = packets[-1]

                if latestPacket["rgb"] is not None:
                    self.frameRgb = latestPacket["rgb"].getCvFrame()
                    # self.frameRgb = cv2.cvtColor(self.frameRgb, cv2.COLOR_BGR2RGB)
                    self.frame.emit(self.frameRgb)

                if latestPacket["depth"] is not None:
                    self.frameDepth16 = latestPacket["depth"].getFrame()
                    # self.frameDepth16 = fill_bg_with_fg(self.frameDepth16.astype(np.int32)).astype(np.uint16)
                    self.frameDepth8 = self.frameDepth16.copy()
                    self.frameDepth8[self.frameDepth8 > 10000] = 0
                    self.frameDepth8 = cv2.normalize(self.frameDepth8, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    self.frameDepth8 = cv2.applyColorMap(self.frameDepth8, cv2.COLORMAP_JET)
                    self.frameDepth8 = np.ascontiguousarray(self.frameDepth8)
                    self.frame_depth8.emit(self.frameDepth8)
                    self.frame_depth16.emit(self.frameDepth16)

# camera = CameraWorker()

# ---------------------------------------------------------- #
##############################################################
####################### User Interface #######################
##############################################################

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.worker = CameraWorker()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 830+200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.save_color_label_directory = None
        self.save_label_directory = None

        self.rgb_photo = QtWidgets.QLabel(self.centralwidget)
        self.rgb_photo.setGeometry(QtCore.QRect(10, 10, 600, 340))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.rgb_photo.setFont(font)
        self.rgb_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.rgb_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.rgb_photo.setLineWidth(5)
        self.rgb_photo.setMidLineWidth(0)
        self.rgb_photo.setObjectName("rgb_photo")

        self.depth_photo = QtWidgets.QLabel(self.centralwidget)
        self.depth_photo.setGeometry(QtCore.QRect(650+10, 10, 600, 340))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.depth_photo.setFont(font)
        self.depth_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.depth_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.depth_photo.setLineWidth(5)
        self.depth_photo.setMidLineWidth(0)
        self.depth_photo.setObjectName("depth_photo")

        self.predict_photo = QtWidgets.QLabel(self.centralwidget)
        self.predict_photo.setGeometry(QtCore.QRect(650+10-330, 10+390-5, 600, 340))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.predict_photo.setFont(font)
        self.predict_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.predict_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.predict_photo.setLineWidth(5)
        self.predict_photo.setMidLineWidth(0)
        self.predict_photo.setObjectName("predict_photo")

        self.streaming = QtWidgets.QPushButton(self.centralwidget)
        self.streaming.isCheckable()
        self.streaming.setGeometry(QtCore.QRect(10+800-300-300, 750+30, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.streaming.setFont(font)
        self.streaming.setObjectName("streaming")
        self.streaming.clicked.connect(self.stream_rgb_function)

        self.generate = QtWidgets.QPushButton(self.centralwidget)
        self.generate.setGeometry(QtCore.QRect(10+800-300, 750+30, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.generate.setFont(font)
        self.generate.setObjectName("generate")
        self.generate.clicked.connect(self.predict_function)

        self.stop = QtWidgets.QPushButton(self.centralwidget)
        self.stop.setGeometry(QtCore.QRect(10+800+300-300, 750+30, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.stop.setFont(font)
        self.stop.setObjectName("generate")
        self.stop.clicked.connect(self.stop_function)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.rgb_photo.setText(_translate("MainWindow", "                   RGB INPUT"))
        self.rgb_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.depth_photo.setText(_translate("MainWindow", "                 DEPTH INPUT"))
        self.depth_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.predict_photo.setText(_translate("MainWindow", "             PREDICTED OUTPUT"))
        self.predict_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.streaming.setText(_translate("MainWindow", "STREAM"))
        self.streaming.setStyleSheet("color: white; font: 20pt Microsoft YaHei UI")

        self.generate.setText(_translate("MainWindow", "PREDICT"))
        self.generate.setStyleSheet("color: white; font: 20pt Microsoft YaHei UI")

        self.stop.setText(_translate("MainWindow", "STOP"))
        self.stop.setStyleSheet("color: white; font: 20pt Microsoft YaHei UI")

    def stream_rgb_function(self):
        self.worker.start()
        self.worker.frame.connect(self.get_rgb_frame)
        self.worker.frame_depth8.connect(self.get_depth8_frame)
        self.worker.frame_depth16.connect(self.get_depth16_frame)

    def stop_function(self):
        self.worker.frame.disconnect()

    def predict_function(self):
        # self.worker.start()
        self.worker.frame.connect(self.predict_frame)

    def get_rgb_frame(self, frame):
        self.rgb_frame = frame
        self.rgb_frame = cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2RGB)
        # self.rgb_out.write(cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2RGB))
        # rgb_map = QImage(self.rgb_frame.data, self.rgb_frame.shape[1], self.rgb_frame.shape[0], self.rgb_frame.shape[1]*3, QImage.Format_RGB888)
        # rgb_map_pix = QPixmap(rgb_map).scaledToWidth(580)
        # self.rgb_photo.setPixmap(rgb_map_pix)
        # self.rgb_photo.setAlignment(Qt.AlignCenter)

    def get_depth8_frame(self, frame):
        # self.depth_out.write(cv2.cvtColor(self.depth16_frame, cv2.COLOR_BGR2RGB))
        # depth_map = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888)
        # depth_map_pix = QPixmap(depth_map).scaledToWidth(580)
        # self.depth_photo.setPixmap(depth_map_pix)
        # self.depth_photo.setAlignment(Qt.AlignCenter)
        pass

    def get_depth16_frame(self, frame):
        self.depth16_frame = frame

    def predict_frame(self):
        global prev_frame_time, new_frame_time

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX

        rgb_image = self.rgb_frame
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        distance_image = self.depth16_frame
        depth8_image = distance_image.copy() 
        depth8_image[depth8_image >= 8000] = 0
        depth8_image = cv2.normalize(depth8_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Converts array to tensor
        rgb_tensor = T.ToTensor()(rgb_image)
        depth8_tensor = T.ToTensor()(depth8_image[:, :, np.newaxis])

        # Resizes transform tensor
        rgb_tensor    = resize_transform(rgb_tensor)
        depth8_tensor = resize_transform(depth8_tensor)

        # # Adds to dict to prepare dataset
        # dataset = dict()
        # dataset["rgb_image"] = rgb_tensor.unsqueeze(0)
        # dataset["depth_image"] = depth8_tensor.unsqueeze(0)
        
        # # Generates output
        # with torch.no_grad():
        #     model.set_input(dataset)
        #     model.forward()
        #     _, pred = torch.max(model.output.data.cpu(), 1)
        #     pred = pred[0].float().detach().int().numpy()

        # # Label map
        # label_map = pred.copy()
        # label_map = cv2.resize(label_map.astype(np.uint8), output_shape, interpolation=cv2.INTER_AREA)

        # # Filters out small objects
        # temp = np.zeros(label_map.shape, np.uint8)
        # temp[label_map == 2] = 1

        # _, label_filter_binary = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV)
        # contours, _ = cv2.findContours(label_filter_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # filter_contours = [contours[i] for i in range(len(contours)) if (cv2.contourArea(contours[i]) > 1000) and (cv2.contourArea(contours[i]) <= 150000)]
        # new_filter_contours = []
        # for i in range(len(filter_contours)):
        #     rect = cv2.boundingRect(filter_contours[i])
        #     x, y, w, h = rect
        #     bb = label_map[y:y+h, x:x+w]
        #     elements = np.unique(bb, return_counts=True)[0].tolist()
        #     if 1 in elements:
        #         if np.unique(bb, return_counts=True)[1][1] >= 300:
        #             new_filter_contours.append(filter_contours[i])

        # mask = np.zeros(label_map.shape, np.uint8)
        # cv2.drawContours(mask, new_filter_contours, -1, 255, cv2.FILLED)
        # mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # label_map[(label_map == 2)] = 0
        # label_map[mask == 1] = 2

        # # Defines alpha value for blended image
        # blend_scale = 0.5
        # colorized_label = np.array(rgb_image).copy()
        # for i in range(3):
        #     channel = colorized_label[:, :, i]
        #     channel[label_map==0] = blend_scale*label_color[2, i] + (1-blend_scale)*channel[label_map==0]
        #     channel[label_map==1] = blend_scale*label_color[1, i] + (1-blend_scale)*channel[label_map==1]
        #     channel[label_map==2] = blend_scale*label_color[0, i] + (1-blend_scale)*channel[label_map==2]
        #     colorized_label[..., i] = channel

        # for i in range(len(new_filter_contours)):
        #     rect = cv2.boundingRect(new_filter_contours[i])
        #     x, y, w, h = rect
        #     bb = label_map[y:y+h, x:x+w]
        #     elements = np.unique(bb, return_counts=True)[0].tolist()
        #     x_center, y_center = int(x + w/2), int(y + h/2)
        #     delta = 3
        #     region = distance_image[y_center-delta:y_center+delta, x_center-delta:x_center+delta]
        #     region[region == 0] == np.nan
        #     distance = np.round(np.max(region))
        #     # Draws to output image the result
        #     cv2.circle(colorized_label, (x_center, y_center), 5, (255, 255, 255), 5)
        #     cv2.rectangle(colorized_label, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #     cv2.putText(colorized_label, "X: "+str(round(x-640, 2))+"mm", (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.4e-3*720, (255, 0, 0), 2)
        #     cv2.putText(colorized_label, "Y: "+str(round(360-y, 2))+"mm", (x+10, y+75), cv2.FONT_HERSHEY_SIMPLEX, 1.4e-3*720, (255, 0, 0), 2)
        #     cv2.putText(colorized_label, "Z: "+str(distance)+"mm", (x+10, y+120), cv2.FONT_HERSHEY_SIMPLEX, 1.4e-3*720, (255, 0, 0), 2)


        # # time when we finish processing for this frame
        # new_frame_time = time.time()

        # fps = 1/(new_frame_time-prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = str(int(fps))
        # cv2.putText(colorized_label, "FPS:"+fps, (10, 50), font, 1.5, (100, 255, 0), 3, cv2.LINE_AA)

        # color_anomaly_map = QImage(colorized_label.data, colorized_label.shape[1], colorized_label.shape[0], colorized_label.shape[1]*3, QImage.Format_RGB888)
        # color_anomaly_map_pix = QPixmap(color_anomaly_map).scaledToWidth(580)
        # self.predict_photo.setPixmap(color_anomaly_map_pix)
        # self.predict_photo.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setGeometry(0+100, 0+50, 1420-150, 840+100)
    MainWindow.setWindowTitle("Niko V1.0")
    qtmodern.styles.dark(app)
    mw = qtmodern.windows.ModernWindow(MainWindow)
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    mw.show()
    sys.exit(app.exec_())