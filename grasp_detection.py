import cv2
import datetime
import darknet
import random
import time
import numpy as np
from queue import Queue
from threading import Thread, enumerate


weights = "yolo_data/yolov4-obj_final.weights"
config = "yolo_data/yolov4-obj.cfg"
classes = "yolo_data/obj.names"
data = "yolo_data/obj.data"
thresh = 0.7
show_coordinates = True

save_path_columnar = "/home/user/shape_detection/columnar/"
save_path_long = "/home/user/shape_detection/long/"
save_path_circle = "/home/user/shape_detection/circle/"
save_path_blade = "/home/user/shape_detection/blade/"
save_bgm = "/home/user/shape_detection/bgm/"

curr_time = datetime.datetime.now()

# Load YOLO
network, class_names, class_colors = darknet.load_network(config,data,weights,batch_size=1)

cap = cv2.VideoCapture(0)
cap.set(3,1600)
cap.set(4,896)
fps = 60

while cap.isOpened():

    i = -1
    a = 0
    while(True):
    
        ret, frame = cap.read()
        if not ret:
            break

        if i == -1:
            i += 1
            cv2.imwrite(save_bgm+str(i) +'.jpg',frame)
            

        width = frame.shape[1]
        height = frame.shape[0]

        t_prev = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))

        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) 
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.print_detections(detections, show_coordinates)
        # label,confidence,x,y,w,h
     
        darknet.free_image(darknet_image)

        #draw bounding box
        image = darknet.draw_boxes(detections, frame_resized, class_colors)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

        fps = int(1/(time.time()-t_prev))

        # cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
        # cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        
        if  len(detections) != 0:
            if int(float(detections[0][1])) >= 90:
                if detections[0][0] == "long":
                    cv2.imwrite(save_path_long+str(a) +'.jpg',image)
                    a += 1
            
                elif detections[0][0] == "circle":
                    cv2.imwrite(save_path_circle+str(a) +'.jpg',image)
                    a += 1
            
                elif detections[0][0] == "columnar":
                    cv2.imwrite(save_path_columnar+str(a) +'.jpg',image)
                    a += 1

                elif detections[0][0] == "blade":
                    cv2.imwrite(save_path_blade+str(a) +'.jpg',image)
                    a += 1

        
        k = cv2.waitKey(1)

        # if k == 32:
        #     print("save~background")
        #     cv2.imwrite('/home/user/shape_detection/'+str(a) +'.jpg',image)
        #     a+=1
        if k == 27:
            break
        cv2.imshow("win_title", image)
        
    cv2.destroyAllWindows()
    cap.release()





# net = cv2.dnn.readNet(weights, config)

# with open(classes, "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = net.getUnconnectedOutLayersNames()
# colors = np.random.uniform(0, 255, size=(len(classes), 3))







