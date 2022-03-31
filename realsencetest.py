import cv2
import numpy as np
import pyrealsense2 as rs
import time

def get_center_depth(depth):
    # Get the depth frame's dimensions
    width = depth.get_width()
    height = depth.get_height()

    center_x = int(width / 2)
    center_y = int(height / 2)

    print(center_x, " ", center_y)
    dis_center = round(depth.get_distance(center_x, center_y)*100, 2)
    print("The camera is facing an object ", dis_center, " cm away.")
    return dis_center, (center_x, center_y)

if __name__ == '__main__':
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Configure and start the pipeline
    pipeline.start(config)

    while True:
        start = time.time()
        # Block program until frames arrive
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        print(type(depth_image))

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.imwrite('output.jpg',color_image)
        dis_center, center_coordinate = get_center_depth(depth_frame)
        

        print("color_image:", color_image.shape)
        cv2.circle(color_image, center_coordinate, 5, (0,0,255), 0)
        cv2.imshow("color_image", color_image)
        # cv2.imshow("depth_image", dis_center)
        
        # print("FPS:", 1/(time.time()-start), "/s")

        # press Esc to stop
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            pipeline.stop()
            break

    cv2.destroyAllWindows()
