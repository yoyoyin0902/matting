import pyzed.sl as sl
import cv2
import numpy as np

class CameraZed2:
    def __init__(self,resolution=None,fps=30,depthMode = None):
        self.zed = sl.Camera()
        self.input_type = sl.InputType()
        self.init_params = sl.InitParameters(input_t=self.input_type)
        # 设置分辨率
        if resolution == "2K":
            self.init_params.camera_resolution = sl.RESOLUTION.HD2K
        elif resolution == "1080":
            self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:  # 默认
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = fps  # 设置帧率
        # 设置获取深度信息的模式
        if depthMode == "PERFORMANCE":
            self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        elif depthMode == "QUALITY":
            self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        else:
            self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # 单位毫米
        # 打开相机
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.STANDARD
        self.savepath = ''  # 标定图像保存的路径

    def grab_imgs(self):  # 捕获左右图像用于相机标定（文件夹自动创建）
        img_l = sl.Mat()
        img_r = sl.Mat()
        num = 0
        num1 = 1
        # 自动创建保存文件夹（分别存放左图和右图）
        import time
        name = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.savepath_L = "/home/user/cali/left/"
        self.savepath_R = "/home/user/cali/right/"
        # os.makedirs(self.savepath_L,exist_ok=True)
        # os.makedirs(self.savepath_R,exist_ok=True)


        while True:
            if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(img_l,sl.VIEW.LEFT)
                self.img_l = img_l.get_data()
                self.zed.retrieve_image(img_r,sl.VIEW.RIGHT)
                self.img_r = img_r.get_data()
                view = np.concatenate((self.img_l,self.img_r),axis=1)
                cv2.imshow('View',cv2.resize(view,(1920,540)))
                key = cv2.waitKey(1)
                if key & 0xFF == ord('s'):  # 按S同时保存左右图像
                    savePath_L = self.savepath_L + str(num)+".jpg"
                    print(savePath_L)
                    cv2.imwrite(savePath_L, self.img_l)
                    savePath_R = self.savepath_R + str(num1)+".jpg"
                    cv2.imwrite(savePath_R, self.img_r)
                    num +=2
                    num1 +=2
                if key & 0xFF == 27:  # 按esc退出视图窗口
                    break

if __name__ == "__main__":
    cam = CameraZed2(resolution='1080',fps=30)
    cam.grab_imgs()  # 获取标定图像（左、右图）
