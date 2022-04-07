# This Python file uses the following encoding: utf-8
import time
import numpy as np
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
import depthai as dai


class OakDCamera(QThread):
    signals = pyqtSignal(list)

    def __init__(self, fps, conf_thread, max_depth):
        super(OakDCamera, self).__init__()
        self._fps = fps
        self._conf_thresh = conf_thread
        self._max_depth = max_depth
        self._device = None
        self._rgb = None
        self._disp = None
        self._depth = None
        self._is_killed = False
        self._is_paused = False
        self._is_connected = False
        self._mutex = QMutex()
        self._init_camera()

    def _init_camera(self):
        # Define Oak-D pipeline
        pipeline = dai.Pipeline()
        # Define RGB source
        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb_cam.initialControl.setManualFocus(135)
        rgb_cam.setIspScale(2, 3)
        rgb_cam.setFps(self._fps)
        # Define mono sources
        # Define left source
        left_cam = pipeline.createMonoCamera()
        left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        left_cam.setFps(self._fps)
        # Define right source
        right_cam = pipeline.createMonoCamera()
        right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        right_cam.setFps(self._fps)
        # Define stereo source
        stereo_depth = pipeline.createStereoDepth()
        stereo_depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo_depth.setRectifyEdgeFillColor(0)
        stereo_depth.setConfidenceThreshold(self._conf_thresh)
        stereo_depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        # stereo_depth.initialConfig.setBilateralFilterSigma(250)
        stereo_depth.setLeftRightCheck(True)
        # Define outputs
        rgb_out = pipeline.createXLinkOut()
        rgb_out.setStreamName('rgb')
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName('depth')
        # Link sources to outputs
        rgb_cam.isp.link(rgb_out.input)
        left_cam.out.link(stereo_depth.left)
        right_cam.out.link(stereo_depth.right)
        stereo_depth.depth.link(depth_out.input)
        self._device = dai.Device(pipeline)
        self._is_connected = self._device.isPipelineRunning()

    def run(self):
        self._device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
        self._device.getOutputQueue(name='depth', maxSize=4, blocking=False)
        while self._is_connected:
            rgb_packets = self._device.getOutputQueue('rgb').tryGetAll()
            depth_packets = self._device.getOutputQueue('depth').tryGetAll()
            if len(rgb_packets) > 0:
                self._rgb = rgb_packets[-1].getCvFrame()
            if len(depth_packets) > 0:
                self._depth = depth_packets[-1].getFrame()
                # self._depth[self._depth > self._max_depth] = 0
            self._mutex.lock()
            if (not self._is_paused) and (self._rgb is not None) and (self._depth is not None):
                self.signals.emit([self._rgb, self._depth])
            self._mutex.unlock()
            time.sleep(0.05)
        self._device.close()

    def update_max_depth(self, max_depth):
        self._mutex.lock()
        self._max_depth = max_depth
        self._mutex.unlock()

    def pause(self):
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._is_paused = False
        self._mutex.unlock()

    def close(self):
        self._mutex.lock()
        self._is_connected = False
        self.quit()
        self.wait()
        self._mutex.unlock()

    def is_paused(self):
        return self._is_paused

    def is_connected(self):
        return self._is_connected

    def get_camera_info(self):
        calib_data = self._device.readCalibration()
        rgb_intrinsics = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
        lr_extrinsics = np.array(calib_data.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
        return rgb_intrinsics, lr_extrinsics[:-1]

if __name__ == '__main__':
    run()
