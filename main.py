import sys
import time
from pathlib import Path
import torchvision.transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QDir
import numpy as np
import cv2
from form import Ui_OakDDetector
from oakd_camera import OakDCamera
from fusenet import load_fusenet_model, transforms, predict, contour_filter
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import matplotlib.pyplot as plt


class OakDDetector(QMainWindow, Ui_OakDDetector):
    def __init__(self):
        super(OakDDetector, self).__init__()
        self._fps = 5
        self._conf_thresh = 245
        self._max_depth = 10000
        self._is_streaming = False
        self._is_connected = False
        self._is_recorded = False
        self.rgb = None
        self.depth = None
        self.pred = None
        self.writer = None
        self._save_root = None
        # used to record the time when we processed last frame
        self.prev_frame_time = time.time()
        # used to record the time at which we processed current frame
        self.new_frame_time = time.time()
        # Defines color for labels such as 0, 1, 2
        self.label_color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        self._camera = OakDCamera(self._fps, self._conf_thresh, self._max_depth)
        self.model = load_fusenet_model()

        self._load_ui()

    def _load_ui(self):
        # Load *.ui file
        self._ui = Ui_OakDDetector()
        self._ui.setupUi(self)
        # Setup default for ui
        self._ui.streamButton.setEnabled(True)
        self._ui.stopButton.setEnabled(False)
        self._ui.predButton.setEnabled(True)
        self._ui.recordButton.setEnabled(True)
        # Connect Qt objects to methods
        self._ui.streamButton.clicked.connect(self._stream_btn_clicked)
        self._ui.stopButton.clicked.connect(self._stop_btn_clicked)
        self._ui.predButton.clicked.connect(self._pred_btn_clicked)
        self._ui.recordButton.clicked.connect(self._record_btn_clicked)
        self._ui.browseButton.clicked.connect(self._browse_btn_clicked)

    def _stream_btn_clicked(self):
        if self._camera.is_connected():
            if self._camera.is_paused():
                self._camera.resume()
            self._camera.signals.connect(self._view_data)
            self._camera.start(QThread.LowPriority)
            # Lock stream button and activate stop button
            self._ui.streamButton.setEnabled(not self._ui.streamButton.isEnabled())
            self._ui.stopButton.setEnabled(not self._ui.stopButton.isEnabled())
            self._is_streaming = True

    def _stop_btn_clicked(self):
        if self._is_recorded:
            msg = QMessageBox(text='Please stop capture before stopping stream!')
            msg.exec()
            return
        self._camera.signals.disconnect(self._view_data)
        self._camera.signals.disconnect(self._predict_data)
        self._camera.pause()
        self._ui.rgbLabel.clear()
        self._ui.depthLabel.clear()
        self._ui.predLabel.clear()
        self._ui.rgbLabel.setText('RGB')
        self._ui.depthLabel.setText('DEPTH')
        self._ui.predLabel.setText('PREDICT')
        self._ui.streamButton.setEnabled(not self._ui.streamButton.isEnabled())
        self._ui.stopButton.setEnabled(not self._ui.stopButton.isEnabled())
        self._ui.predButton.setEnabled(not self._ui.predButton.isEnabled())
        self._is_streaming = False

    def _pred_btn_clicked(self):
        if not self._is_streaming:
            msg = QMessageBox(text='Please press \'Stream\' before predicting!')
            msg.exec()
            return
        self._camera.signals.connect(self._predict_data)
        self._ui.predButton.setEnabled(not self._ui.predButton.isEnabled())

    def _record_btn_clicked(self):
        if not self._is_recorded:
            # if self._camera.is_paused() or not self._is_streaming:
            #     msg = QMessageBox(text='Please stream camera!')
            #     msg.exec()
            #     return
            if self._save_root is None or self._save_root == '':
                msg = QMessageBox(text='The saving directory is empty!')
                msg.exec()
                return
            self._camera.start(QThread.LowPriority)
            self._camera.signals.connect(self._predict_data)
            self._camera.signals.connect(self._record_data)
            self._is_recorded = True
            self._ui.recordButton.setText('Stop')
        else:
            self._camera.signals.disconnect(self._record_data)
            self._camera.signals.disconnect(self._predict_data)
            self._is_recorded = False
            self._ui.recordButton.setText('Record')
            self.writer.release()
        self._ui.browseButton.setEnabled(not self._ui.browseButton.isEnabled())
        self._ui.browseLineEdit.setEnabled(not self._ui.browseLineEdit.isEnabled())

    def _browse_btn_clicked(self):
        self._save_root = QFileDialog.getExistingDirectory(self, 'Select a directory', self._save_root)
        if self._save_root:
            self._save_root = QDir.toNativeSeparators(self._save_root)
            self._ui.browseLineEdit.setText(self._save_root)
        self.writer = cv2.VideoWriter(self._save_root + '/record.avi',
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                      15,
                                      (1280 * 3, 720))

    def _view_data(self, data):
        self._update_view_label(data[0], self._ui.rgbLabel, 'rgb')
        self._update_view_label(data[1], self._ui.depthLabel, 'depth')
        pass

    def _predict_data(self, data):
        # font which we will be using to display FPS
        rgb = data[0].copy()
        depth = cv2.normalize(data[1], None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        rgb = transforms(rgb)
        depth = transforms(depth[:, :, np.newaxis])
        pred = predict(self.model, rgb, depth)
        pred = pred.astype(np.uint8)
        # Filters out small objects
        mask = np.zeros(pred.shape, np.uint8)
        mask[pred == 2] = 1
        _, label_filter_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(label_filter_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contours[i] for i in range(len(contours)) if
                    (cv2.contourArea(contours[i]) > 200) and (cv2.contourArea(contours[i]) <= 50000)]
        filtered_contours = contour_filter(pred, contours, 30)
        mask = np.zeros(pred.shape, np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 1, cv2.FILLED)
        # mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
        pred[(pred == 2)] = 0
        pred[mask == 1] = 2
        # Defines alpha value for blended image
        # Label map
        rgb = np.array(torchvision.transforms.ToPILImage()(rgb).convert('RGB'))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        blend_scale = 0.5
        for i in range(3):
            rgb[:, :, i][pred == 0] = \
                blend_scale * self.label_color[2, i] + (1 - blend_scale) * rgb[:, :, i][pred == 0]
            rgb[:, :, i][pred == 1] = \
                blend_scale * self.label_color[1, i] + (1 - blend_scale) * rgb[:, :, i][pred == 1]
            rgb[:, :, i][pred == 2] = \
                blend_scale * self.label_color[0, i] + (1 - blend_scale) * rgb[:, :, i][pred == 2]
        rgb = self._draw_result(rgb, data[1], filtered_contours)
        # key, route = self._draw_path(pred)
        # if key != None:
            # rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
            # for i in range(route.shape[0]):
            #     cv2.circle(rgb, (int(route[i, 0]), int(route[i, 1])), 2, (255, 0, 0), 2)
            # tic = time.time()
            # path = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"path_{tic}.png", path)
        # time when we finish processing for this frame
        self.new_frame_time = time.time()
        self._fps = int(1. / (self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time
        cv2.putText(rgb,
                    'FPS:' + str(self._fps),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        color_anomaly_map = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        self._ui.predLabel.setPixmap(QPixmap(color_anomaly_map))
        self.map = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)
        self.map = cv2.cvtColor(self.map, cv2.COLOR_RGB2BGR)

    def _record_data(self, data):
        self.rgb = data[0]
        self.depth = cv2.normalize(data[1], None, 255, 0, cv2.NORM_INF, cv2.CV_8U)
        self.depth = cv2.applyColorMap(self.depth, cv2.COLORMAP_JET)
        if (self.map is not None) and (self.rgb is not None) and (self.depth is not None):
            data = np.concatenate([self.rgb, self.depth, self.map], axis=1)
            self.writer.write(data)

    @staticmethod
    def _update_view_label(img, label, mode='rgb'):
        if mode == 'disp':
            img = (img * (255 / 96)).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        elif mode == 'depth':
            img = cv2.normalize(img, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        img = cv2.resize(img, (label.width(), label.height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        img = QPixmap(img)
        label.setPixmap(img)

    def _draw_result(self, rgb, depth, contours):
        rgb = cv2.resize(rgb, (self._ui.predLabel.width(), self._ui.predLabel.height()), interpolation=cv2.INTER_LINEAR)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            x_center, y_center = int(x + w / 2), int(y + h / 2)
            delta = 3
            region = depth[3 * y_center - delta: 3 * y_center + delta, 4 * x_center - delta: 4 * x_center + delta]
            region[region == 0] == np.nan
            distance = np.round(np.max(region))
            # Draws to output image the result
            cv2.circle(rgb, (2 * x_center, int(1.5 * y_center)), 2, (255, 255, 255), 2)
            cv2.rectangle(rgb, (2 * x, int(1.5 * y)), (2 * (x + w), int(1.5 * (y + h))), (255, 0, 0), 2)
            cv2.putText(rgb,
                        'X: ' + str(round(x - 640, 2)) + 'mm',
                        (2 * x + 50, int(1.5 * y) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(rgb,
                        'Y: ' + str(round(360 - y, 2)) + 'mm',
                        (2 * x + 50, int(1.5 * y) + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(rgb,
                        'Z: ' + str(distance) + 'mm',
                        (2 * x + 50, int(1.5 * y) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            
        return rgb
    
    def _draw_path(self, pred):
        # Binary map
        binary_map = np.ones(pred.shape)
        binary_map[pred == 2] = 0
        binary_map[pred == 0] = 0
        binary_map = cv2.resize(binary_map, (640, 360))
        
        h, w= binary_map.shape[:2]
        alpha = 1/24
        T = w * alpha

        dct = {}
        key = None
        i = 0
        for row in range(h):
            dct[row] = list()
            value = 0
            temp = []
            for col in range(w):
                if binary_map[row, col] == 1 and value == 0:
                    temp.append([row, col])
                    value = 1
                elif binary_map[row, col] == 0 and len(temp) == 1 and value == 1:
                    temp.append([row, col-1])
                if len(temp) == 2 and value == 1:
                    value = 0
                    if abs(temp[0][1] - temp[1][1]) >= T:
                        dct[row].append(temp)
                        temp = []
                        if i == 0:
                            key = row
                            i += 1
            if len(dct[row]) == 0:
                del dct[row]

        if key != None:
            des_joint = None
            farest_points = dct[key]
            if len(farest_points) > 1:
                longest = abs(farest_points[0][1][1] - farest_points[0][0][1])
                for i in range(len(farest_points)):
                    distance = abs(farest_points[i][1][1] - farest_points[i][0][1])
                    if distance >= longest:
                        des_joint = farest_points[i]
                    else:
                        des_joint = farest_points[0]
                        
            elif len(farest_points) == 1:
                des_joint = farest_points[0]
            
            destination = [int((des_joint[0][1] + des_joint[1][1]) / 2), int(des_joint[0][0])]
            
            d = bwdist(binary_map == 1)
            
            # Rescale and transform distance
            d2 = (d/100.) + 1
            d0 = 2
            nu = 800
            repulsive = nu*((1/d2 - 1/d0)**2)
            repulsive[d2 > d0] = 0
            
            [x, y] = np.meshgrid(np.arange(w), np.arange(h))
            goal = destination
            start = [w//2, h-20]
            xi = 1/700
            attractive = xi * ((x - goal[0])**2 + (y - goal[1])**2)
            
            f = attractive + repulsive
            
            route = self._gradientBasedPlanner(f, start, goal, 700)
        else:
            route = [0, 0]
            
        return key, route
                
    def _gradientBasedPlanner(self, f, start_coords, end_coords, max_its):
        [gy, gx] = np.gradient(-f)
        route = np.vstack([np.array(start_coords), np.array(start_coords)])
        for i in range(max_its):
            current_point = route[-1, :]
            if sum(abs(current_point - end_coords)) < 5.0:
                break
            ix = int(round(current_point[1]))
            iy = int(round(current_point[0]))
            # print(ix, iy)
            if ix >= 360:
                ix = 359
            vx = gx[ix, iy]
            vy = gy[ix, iy]
            dt = 1/np.linalg.norm([vx, vy])
            next_point = current_point + dt * np.array([vx, vy])
            route = np.vstack([route, next_point])
        route = route[1:, :]
        return route
        

    def closeEvent(self, event):
        if self._is_recorded:
            msg = QMessageBox(text='Please stop capturing!')
            msg.exec()
            event.ignore()
        else:
            self._camera.close()
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Force the style to be the same on all OS
    app.setStyle('Fusion')
    #
    widget = OakDDetector()
    widget.show()
    sys.exit(app.exec())
