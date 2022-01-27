import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt

 
# def pretreatment(ima):
#     ima=ima.convert('L')         #转化为灰度图像
#     im=np.array(ima)        #转化为二维数组
#     for i in range(im.shape[0]):#转化为二值矩阵
#         for j in range(im.shape[1]):
#             if im[i,j]==75 or im[i,j]==76:
#                 im[i,j]=1
#             else:
#                 im[i,j]=0
#     return im
# # ima=Image.open('26_img.jpg') #读入图像


# # cv2.waitKey(0)
# # cv2.destroyAllWindows()	

# # im=pretreatment(ima)  #调用图像预处理函数
# # for i in im:
# #     print(i)


# if __name__ =='__main__':
# 	ima1 = cv2.imread('26_img.jpg')
#     cv2.imshow('test',ima1) 

#     ima=Image.open('26_img.jpg') 
#     im=pretreatment(ima)  
#     for i in im:
#         print(i)
    

# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()	

import sys, argparse, numpy
import cv2

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 480
    height_new = 270
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def main():
    parser = argparse.ArgumentParser(description='images')
    parser.add_argument('--src',type = str, required=False,default = "/home/user/matting/area_calculate/26_img.jpg" ,help="the source image")
    parser.add_argument('--dest',type = str, required = False,default="/home/user/matting/area_calculate/smile.txt" ,help="the dest text")
    args = parser.parse_args()
    try: 
        img = cv2.imread(args.src,2)
        img_new = img_resize(img)
        print(img_new.shape)
        
        _, bw_img = cv2.threshold(img_new, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary = np.where(bw_img == 255, 1, bw_img)   #0黑 255白
        [r,c] = size(binary)
        with open(args.dest, 'w') as dest:
            dest.write('[\n')
            for r in range(0, len(binary) - 1):
                # dest.write('\t') #縮排
                dest.write(str(binary[r].tolist()))
                dest.write(',\n')
            # dest.write('\t')
            dest.write(str(binary[-1].tolist()))
            dest.write('\n]')
        
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except:
        print('error')




if __name__ == '__main__':                  
    main()
