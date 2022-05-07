import warnings
import numpy as np
from utils import *
import os, cv2, sys, time

from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import qtmodern
import qtmodern.styles, qtmodern.windows
import _warnings
warnings.filterwarnings('ignore')

default_path = "D:/NTUST_MASTER/Thesis/Master_Thesis/ALS-TIP/GMRPD"

def get_odd(x):
    return int(2 * (x/2 + 1) - 1)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 830+200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.save_color_label_directory = None
        self.save_label_directory = None

        self.rgb_photo = QtWidgets.QLabel(self.centralwidget)
        self.rgb_photo.setGeometry(QtCore.QRect(10, 10, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.rgb_photo.setFont(font)
        self.rgb_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.rgb_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.rgb_photo.setLineWidth(5)
        self.rgb_photo.setMidLineWidth(0)
        self.rgb_photo.setObjectName("rgb_photo")

        self.depth_photo = QtWidgets.QLabel(self.centralwidget)
        self.depth_photo.setGeometry(QtCore.QRect(650+10, 10, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.depth_photo.setFont(font)
        self.depth_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.depth_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.depth_photo.setLineWidth(5)
        self.depth_photo.setMidLineWidth(0)
        self.depth_photo.setObjectName("depth_photo")

        self.rgb_anomaly = QtWidgets.QLabel(self.centralwidget)
        self.rgb_anomaly.setGeometry(QtCore.QRect(1310, 10, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.rgb_anomaly.setFont(font)
        self.rgb_anomaly.setFrameShape(QtWidgets.QFrame.Panel)
        self.rgb_anomaly.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.rgb_anomaly.setLineWidth(5)
        self.rgb_anomaly.setMidLineWidth(0)
        self.rgb_anomaly.setObjectName("rgb_anomaly")

        self.depth_anomaly = QtWidgets.QLabel(self.centralwidget)
        # self.depth_anomaly.setGeometry(QtCore.QRect(670, 10, 600, 340))
        self.depth_anomaly.setGeometry(QtCore.QRect(10, 10+400-20-10-10-30, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.depth_anomaly.setFont(font)
        self.depth_anomaly.setFrameShape(QtWidgets.QFrame.Panel)
        self.depth_anomaly.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.depth_anomaly.setLineWidth(5)
        self.depth_anomaly.setMidLineWidth(0)
        self.depth_anomaly.setObjectName("depth_anomaly")        

        self.browseRGBFile = QtWidgets.QPushButton(self.centralwidget)
        self.browseRGBFile.setGeometry(QtCore.QRect(10, 750-60, 160+20+30+20, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.browseRGBFile.setFont(font)
        self.browseRGBFile.setObjectName("browseRGBFile")
        self.browseRGBFile.clicked.connect(self.browse_rgb_file)

        self.browseDepthFile = QtWidgets.QPushButton(self.centralwidget)
        self.browseDepthFile.setGeometry(QtCore.QRect(10, 800-60, 160+20+30+20, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.browseDepthFile.setFont(font)
        self.browseDepthFile.setObjectName("browseRGBFile")
        self.browseRGBFile.clicked.connect(self.browse_depth_file)

        self.saveLabelFile = QtWidgets.QPushButton(self.centralwidget)
        self.saveLabelFile.setGeometry(QtCore.QRect(10, 850-60, 160+20+30+20, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.saveLabelFile.setFont(font)
        self.saveLabelFile.setObjectName("saveColorLabelFile")
        self.saveLabelFile.clicked.connect(self.save_label)          

        self.generate = QtWidgets.QPushButton(self.centralwidget)
        self.generate.setGeometry(QtCore.QRect(10+800, 750-60, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.generate.setFont(font)
        self.generate.setObjectName("generate")
        self.generate.clicked.connect(self.get_label)

        self.drivable_area_photo = QtWidgets.QLabel(self.centralwidget)
        self.drivable_area_photo.setGeometry(QtCore.QRect(660, 10+420-20-20-10-10-30, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.drivable_area_photo.setFont(font)
        self.drivable_area_photo.setFrameShape(QtWidgets.QFrame.Panel)
        self.drivable_area_photo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.drivable_area_photo.setLineWidth(5)
        self.drivable_area_photo.setObjectName("drivable_area_photo")

        self.anomaly_map = QtWidgets.QLabel(self.centralwidget)
        self.anomaly_map.setGeometry(QtCore.QRect(660+650, 10+420-20-20-10-10-30, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.anomaly_map.setFont(font)
        self.anomaly_map.setFrameShape(QtWidgets.QFrame.Panel)
        self.anomaly_map.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.anomaly_map.setLineWidth(5)
        self.anomaly_map.setObjectName("anomaly")

        self.final_map = QtWidgets.QLabel(self.centralwidget)
        self.final_map.setGeometry(QtCore.QRect(660+650, 10+420-20-20-10-10+330-30, 600, 340-30))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.final_map.setFont(font)
        self.final_map.setFrameShape(QtWidgets.QFrame.Panel)
        self.final_map.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.final_map.setLineWidth(5)
        self.final_map.setObjectName("final")

        self.info_tab = QtWidgets.QLabel(self.centralwidget)
        self.info_tab.setGeometry(QtCore.QRect(10+800, 780-60, 600, 100))
        self.info_tab.setText("Ready For Prediction...!")
        self.info_tab.setStyleSheet("color: white; font: 10pt Microsoft YaHei UI")

        self.lineEdit_rgb = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_rgb.setGeometry(QtCore.QRect(10+250, 750-60, 400-30-10-20+100, 31))
        self.lineEdit_rgb.setObjectName("lineEdit")

        self.lineEdit_depth = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_depth.setGeometry(QtCore.QRect(10+250, 800-60, 400-30-10-20+100, 31))
        self.lineEdit_depth.setObjectName("lineEdit")

        # self.lineEdit_mask = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_mask.setGeometry(QtCore.QRect(1470+20+30+20+20, 20+50+50+10+10, 400-30-10-20, 31))
        # self.lineEdit_mask.setObjectName("lineEdit")

        self.lineEdit_label_save = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_label_save.setGeometry(QtCore.QRect(10+250, 850-60, 400-30-10-20+100, 31))
        self.lineEdit_label_save.setObjectName("lineEdit")

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
        self.rgb_photo.setText(_translate("MainWindow", "              RGB INPUT"))
        self.rgb_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.depth_photo.setText(_translate("MainWindow", "            DEPTH INPUT"))
        self.depth_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.rgb_anomaly.setText(_translate("MainWindow", "        RGB ANOMALY MAP"))
        self.rgb_anomaly.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.browseRGBFile.setText(_translate("MainWindow", "Browse RGB File"))
        self.browseRGBFile.setStyleSheet("color: white; font: 10pt Microsoft YaHei UI")

        self.browseDepthFile.setText(_translate("MainWindow", "Browse Depth File"))
        self.browseDepthFile.setStyleSheet("color: white; font: 10pt Microsoft YaHei UI")

        # self.browseMaskFile.setText(_translate("MainWindow", "Browse Mask File"))
        # self.browseMaskFile.setStyleSheet("color: white; font: 10pt Microsoft YaHei UI")  

        self.saveLabelFile.setText(_translate("MainWindow", "Label Saving Folder"))
        self.saveLabelFile.setStyleSheet("color: white; font: 10pt Microsoft YaHei UI")  

        self.generate.setText(_translate("MainWindow", "PREDICT"))
        self.generate.setStyleSheet("color: white; font: 20pt Microsoft YaHei UI")
        
        self.depth_anomaly.setText(_translate("MainWindow", "    DEPTH ANOMALY MAP"))
        self.depth_anomaly.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.drivable_area_photo.setText(_translate("MainWindow", "      DRIVABLE AREA MAP"))
        self.drivable_area_photo.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.anomaly_map.setText(_translate("MainWindow", "            ANOMALY MAP"))
        self.anomaly_map.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

        self.final_map.setText(_translate("MainWindow", "               LABEL MAP"))
        self.final_map.setStyleSheet("color: white; font: 25pt Microsoft YaHei UI")

    def browse_rgb_file(self):
        self.rgb_directory = QtWidgets.QFileDialog.getOpenFileName(None, "Browse File", default_path, "PNG (*.PNG *.png")[0]
        self.rgb_image_name = self.rgb_directory.split("/")[-1]
        self.rgb_image = QImage(self.rgb_directory)
        # self.rgb_image = self.rgb_image.copy(0, 80, 1280, 640)
        self.rgb_image_resized = self.rgb_image.scaledToWidth(500)
        pixmap = QPixmap(QPixmap.fromImage(self.rgb_image_resized))
        self.rgb_photo.setPixmap(pixmap)
        self.rgb_photo.setAlignment(Qt.AlignCenter)
        self.lineEdit_rgb.setText('{}'.format(self.rgb_directory))
        # self.info_tab.setText(f"Loading RGB Image: {self.rgb_image_name}")
        
    def browse_depth_file(self):
        # self.depth_directory = QtWidgets.QFileDialog.getOpenFileName(self.rgb_directory.replace("rgb", "depth_u16"))
        self.depth_directory = self.rgb_directory.replace("rgb", "depth_u16")
        self.depth_image_name = self.depth_directory.split("/")[-1]
        depth_image = cv2.imread(self.depth_directory)
        depth_Qimage = QImage(depth_image.data, 
                             depth_image.shape[1], 
                             depth_image.shape[0], 
                             depth_image.shape[1]*3, 
                             QImage.Format_RGB888)
        # depth_Qimage = depth_Qimage.copy(0, 80, 1280, 640)
        pixmap = QPixmap(depth_Qimage).scaledToWidth(500)
        self.depth_photo.setPixmap(pixmap)
        self.depth_photo.setAlignment(Qt.AlignCenter)
        self.lineEdit_depth.setText('{}'.format(self.depth_directory))
        # self.info_tab.setText(f"Loading Depth Image: {self.depth_image_name}")

    def save_label(self):
        self.save_label_directory = QtWidgets.QFileDialog.getExistingDirectory()
        self.lineEdit_label_save.setText('{}'.format(self.save_label_directory))

    def get_label(self):
        self.info_tab.setText("Generating Label ...")
        # reads input image
        tic = time.time()
        # Reads RGB input image
        rgb_image = cv2.imread(self.rgb_directory)
        # Read Depth input image
        depth_image = cv2.imread(self.depth_directory, -1)
        
        
        # Resizing RGB Image to dimension 480x640
        rgb_image = resizeImage(rgb_image)
        # Define the Intel RealSense Parameters which using for computing disparity
        BASELINE = 55.0871
        FOCAL_LENGTH = 1367.6650

        # Filtering Object which are far than 10m
        depth_image[depth_image > 9000] = 0
        depth_temp = resizeImage(depth_image)
        depth_mask = depth_temp > 9000

        # Computing Disparity Map
        disparity_map = np.zeros(depth_image.shape)
        positive_mask = depth_image != 0
        negative_mask = depth_image == 0
        disparity_map[positive_mask] = np.around(FOCAL_LENGTH * BASELINE / depth_image[positive_mask])
        disparity_map[negative_mask] = np.nan
        disparity_map = resizeImage(disparity_map)
        disparity_map = np.around(disparity_map).astype('int')

        # Computing the V-Disparity Map
        height, width = disparity_map.shape[0], int(np.nanmax(disparity_map) + 1)

        v_disparity = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                v_disparity[i, j] = len(np.argwhere(disparity_map[i, :] == j))
        v_disparity[v_disparity <= 5] = 0

        # Apply Steerable Gaussian Filter Order
        theta = [0, 45, 100]
        v_disparity_steerable = np.zeros((v_disparity.shape[0], v_disparity.shape[1], 3))
        for i, angle in enumerate(theta):
            v_disparity_steerable[:,:,i] = steerGaussFilterOrder2(v_disparity, angle, 3)

        v_disparity_steerable_diff = np.zeros(v_disparity.shape)
        for i in range(v_disparity_steerable.shape[0]):
            for j in range(v_disparity_steerable.shape[1]):
                v_disparity_steerable_diff[i, j] = np.max(v_disparity_steerable[i, j, :]) - np.min(v_disparity_steerable[i, j, :])

        v_disparity_filter = np.zeros(v_disparity.shape)
        threshold = 30
        v_disparity_filter[v_disparity_steerable_diff >= threshold] = 1

        straight_line, status = houghTransform(v_disparity_filter)
        x1, y1, x2, y2 = straight_line
        drivable_initial = np.zeros(disparity_map.shape)
        drivable_threshold = 5
        for i in range(y1, y2):
            d = (x2 - x1)/(y2 - y1)*i + (x1*y2 - x2*y1)/(y2 - y1)
            for j in range(drivable_initial.shape[1]):
                if (disparity_map[i, j] > d - drivable_threshold) and (disparity_map[i, j] < d + drivable_threshold):
                    drivable_initial[i, j] = 1

        drivable_initial = np.uint8(drivable_initial)
        _, drivable_binary = cv2.threshold(drivable_initial, 0, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(drivable_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contours_image = np.zeros((drivable_binary.shape[0], drivable_binary.shape[1], 3))
        filter_contours = [contours[i] for i in range(len(contours)) if 200 <= cv2.contourArea(contours[i]) <= 50000]
        cv2.drawContours(contours_image, filter_contours, -1, (0, 255, 0), 1)

        depth_anomalies = cv2.cvtColor(np.float32(contours_image), cv2.COLOR_BGR2GRAY)
        depth_anomalies[depth_anomalies == 0.0] = 0
        depth_anomalies[depth_anomalies != 0.0] = 1
        depth_anomalies = ndimage.binary_fill_holes(depth_anomalies).astype(np.float32)
        depth_anomalies = medfilt2d(depth_anomalies, 5)

        depth_anomalies = cv2.normalize(depth_anomalies, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        drivable_initial = cv2.normalize(drivable_initial, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        drivable_area = cv2.bitwise_or(depth_anomalies, drivable_initial, mask=None)
        drivable_area = medfilt2d(drivable_area, 5)
        drivable_area = cv2.normalize(drivable_area, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_16UC1)

        rgb_anomalies = anomaliesRGBGenertor(rgb_image)
        
        rgb_anomalies = np.uint8(rgb_anomalies)
        _, rgb_anomalies_binary = cv2.threshold(rgb_anomalies, 0, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(rgb_anomalies_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_image = np.zeros((rgb_anomalies.shape[0], rgb_anomalies.shape[1], 3))
        filter_contours = [contours[i] for i in range(len(contours)) if 200 <= cv2.contourArea(contours[i]) <= 50000]
        cv2.drawContours(contours_image, filter_contours, -1, (0, 255, 0), 1)

        rgb_anomalies = cv2.cvtColor(np.float32(contours_image), cv2.COLOR_BGR2GRAY)
        rgb_anomalies[rgb_anomalies == 0.0] = 0
        rgb_anomalies[rgb_anomalies != 0.0] = 1
        rgb_anomalies = ndimage.binary_fill_holes(rgb_anomalies).astype(np.float32)
        rgb_anomalies = medfilt2d(rgb_anomalies, 5).astype(np.uint8)
        
        depth_anomalies = cv2.normalize(depth_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
        final_anomalies = cv2.bitwise_or(rgb_anomalies, depth_anomalies)
        final_anomalies = cv2.normalize(final_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
        final_anomalies = medfilt2d(final_anomalies, 5)
        
        # Morphological Processing
        k1 = 1/60
        k5 = 1/48  # drivable area
        min_size = np.min(final_anomalies.shape)

        a1 = get_odd(k1 * min_size)
        a5 = get_odd(k5 * min_size)

        kernel_1 = np.ones((a1, a1), np.uint8)
        kernel_5 = np.ones((a5, a5), np.uint8)

        final_anomalies = cv2.morphologyEx(final_anomalies, cv2.MORPH_CLOSE, kernel_1)
        drivable_area = cv2.morphologyEx(drivable_area, cv2.MORPH_CLOSE, kernel_5)

        sslg_label = np.zeros(disparity_map.shape, dtype=np.uint8)
        sslg_label[drivable_area == 1] = 1
        sslg_label[final_anomalies == 1] = 2
        sslg_label[depth_mask] = 0
        toc = time.time()
        # self.info_tab.setText(f"Processing time: {str(round(toc - tic, 2))} sec")

        # Colorize Label Image And Saving
        # RED (255, 0, 0), GREEN (0, 255, 0), BLUE (0, 0, 255)
        label_color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        colorized_label = rgb_image.copy()

        scale = 0.5
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        colorized_label = rgb_image
        for i in range(label_color.shape[0]):
            channel = colorized_label[:, :, i]
            channel[sslg_label == 0] = (1 - scale) * channel[sslg_label == 0] + scale * label_color[2, i]
            channel[sslg_label == 1] = (1 - scale) * channel[sslg_label == 1] + scale * label_color[1, i]
            channel[sslg_label == 2] = (1 - scale) * channel[sslg_label == 2] + scale * label_color[0, i]
            colorized_label[..., i] = channel
            
        # Shows semantic map to UI
        rgb_anomalies = cv2.normalize(rgb_anomalies, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb_anomalies = cv2.applyColorMap(rgb_anomalies, cv2.COLORMAP_BONE)
        color_rgb_anomaly = QImage(rgb_anomalies.data, rgb_anomalies.shape[1], rgb_anomalies.shape[0], rgb_anomalies.shape[1]*3, QImage.Format_RGB888)
        color_rgb_anomaly_pix = QPixmap(color_rgb_anomaly).scaledToWidth(400)
        self.rgb_anomaly.setPixmap(color_rgb_anomaly_pix)
        self.rgb_anomaly.setAlignment(Qt.AlignCenter)
        
        depth_anomalies = cv2.normalize(depth_anomalies, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_anomalies = cv2.applyColorMap(depth_anomalies, cv2.COLORMAP_BONE)
        color_depth_anomaly = QImage(depth_anomalies.data, depth_anomalies.shape[1], depth_anomalies.shape[0], depth_anomalies.shape[1]*3, QImage.Format_RGB888)
        color_depth_anomaly_pix = QPixmap(color_depth_anomaly).scaledToWidth(400)
        self.depth_anomaly.setPixmap(color_depth_anomaly_pix)
        self.depth_anomaly.setAlignment(Qt.AlignCenter)
        
        drivable_area = cv2.normalize(drivable_area, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        drivable_area = cv2.applyColorMap(drivable_area, cv2.COLORMAP_BONE)
        color_drivable_area = QImage(drivable_area.data, drivable_area.shape[1], drivable_area.shape[0], drivable_area.shape[1]*3, QImage.Format_RGB888)
        color_drivable_area_pix = QPixmap(color_drivable_area).scaledToWidth(400)
        self.drivable_area_photo.setPixmap(color_drivable_area_pix)
        self.drivable_area_photo.setAlignment(Qt.AlignCenter)
        
        final_anomalies = cv2.normalize(final_anomalies, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        final_anomalies = cv2.applyColorMap(final_anomalies, cv2.COLORMAP_BONE)
        color_final_anomalies = QImage(final_anomalies.data, final_anomalies.shape[1], final_anomalies.shape[0], final_anomalies.shape[1]*3, QImage.Format_RGB888)
        color_final_anomalies_pix = QPixmap(color_final_anomalies).scaledToWidth(400)
        self.anomaly_map.setPixmap(color_final_anomalies_pix)
        self.anomaly_map.setAlignment(Qt.AlignCenter)
        
        color_final_map = QImage(colorized_label.data, colorized_label.shape[1], colorized_label.shape[0], colorized_label.shape[1]*3, QImage.Format_RGB888)
        color_final_map_pix = QPixmap(color_final_map).scaledToWidth(400)
        self.final_map.setPixmap(color_final_map_pix)
        self.final_map.setAlignment(Qt.AlignCenter)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setGeometry(0, 0, 1920, 840+100+90)
    MainWindow.setWindowTitle("Niko V1.0")

    qtmodern.styles.dark(app)
    mw = qtmodern.windows.ModernWindow(MainWindow)
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    mw.show()
    sys.exit(app.exec_())
