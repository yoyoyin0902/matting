import cv2
import numpy as np

def max2(x):
    m1 = max(x)
    x2 = x.copy()
    x2.remove(m1)
    m2 = max(x2)
    x3 = x2.copy()
    x3.remove(m2)
    m3 = max(x3)
    return m1,m2,m3 

image = cv2.imread('blade_1.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = []
for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))

id = max2(areas)
max_id2 = areas.index(id[1])
cnt = contours[max_id2] #max contours

for c in cnt:
    # 找到边界坐标
    # x, y, w, h = cv2.boundingRect(c)  # ㄋ

    # 找面积最小的矩形
    rect = cv2.minAreaRect(c)
    # 得到最小矩形的坐标
    box = cv2.boxPoints(rect)

    box = np.int0(box)

	# print(rect)
    print(rect)
    cv2.drawContours(image, [box], 0, (255, 0, 0), 3)
    # 计算最小封闭圆的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 换成整数integer
    center = (int(x),int(y))
    radius = int(radius)
    # 画圆
    # cv2.circle(image, center, radius, (0, 255, 0), 2)

# cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
cv2.imshow("img", image)
cv2.imwrite("img_1.jpg", image)
cv2.waitKey(0)


























# from tkinter import *
# from PIL import Image, ImageTk
# import cv2
# import os
# from tkinter import filedialog
# from pygame import mixer
# import requests
# import base64

# class Application(Frame):
#     def __init__(self, master=None):
#         Frame.__init__(self,master,bg = 'white')
#         self.pack(expand=YES,fill=BOTH) #expand参数表示的是容器在整个窗口上，将容器放置在剩余空闲位置上的中央(包括水平和垂直方向)
#                                         #expand=1或者expand=“yes”，表示放置在中央,expand=0或者expand=“no”，表示默认不扩展
#                                         #fill=“x”，表示横向填充,fill=“y”，表示纵向填充,fill=“both”，表示横向和纵向都填充
#     def window_init(self):
#         self.master.title('帅强相机')
#         #width,height=self.master.maxsize()
#         width, height =800,600
#         screenwidth = self.master.winfo_screenwidth()#获取电脑屏幕宽度
#         screenheight = self.master.winfo_screenheight()
#         alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
#         self.master.geometry(alignstr)#设置大小和位置

#     def createWidgets(self):
#         self.fm1 = Frame(self)# 创建控件容器fm1
#         self.titleLabel = Label(self.fm1, text='欢迎使用帅强二号相机',font =('微软雅黑',30),fg = '#3366FF',bg = '#00CCFF',width  = 800) #向fm1中加入组件
#         self.titleLabel.pack()# 将该控件容器加入到窗口中
#         self.fm1.pack(side= TOP,expand = 1)#将该控件放在窗口顶部

#         self.fm2 = Frame(self,bg = 'white')# 创建控件容器fm2
#         self.fm2_left = Frame(self.fm2,bg = 'white')# 创建控件容器fm2_left
#         self.fm2_right = Frame(self.fm2,bg = 'white')# 创建控件容器fm2_right
#         self.fm2_left_top = Frame(self.fm2_left,bg = 'white')# 创建控件容器fm2_left_top
#         self.fm2_left_bottom = Frame(self.fm2_left,bg = 'white')# 创建控件容器fm2_right_bottom
#         self.fm2_right_top = Frame(self.fm2_right, bg='white')  # 创建控件容器fm2_left_top
#         self.fm2_right_bottom = Frame(self.fm2_right, bg='white')  # 创建控件容器fm2_right_bottom

#         self.cameraButton = Button(self.fm2_left_top, text='拍照',bg = '#00CCFF',fg = 'black',
#                                    font = ('微软雅黑',15),width = 10,command=lambda: take_photos(i))#为fm2_left_top添加按钮
#         self.cameraButton.pack(side=LEFT)#将fm2_left按钮放置在fm2的左边
#         #self.predictEntry = Entry(self.fm2_left_top)#为fm2_left_top添加文本框
#         #self.predictEntry.pack(side=LEFT)#文本框靠左
#         self.fm2_left_top.pack(side=TOP)#将fm1_left_top控件放在fm2_left的顶部

#         self.videoButton = Button(self.fm2_left_bottom, text='录像',bg = '#00CCFF',fg = 'black',
#                                   font = ('微软雅黑',15),width = 10, command=lambda: video(k))
#         self.videoButton.pack(side=LEFT)
#         #self.truthEntry = Entry(self.fm2_left_bottom)
#         #self.truthEntry.pack(side=LEFT)
#         self.fm2_left_bottom.pack(side=TOP)

#         self.predictButton = Button(self.fm2_left_top, text='相册', bg='#00CCFF', fg='black',
#                                     font=('微软雅黑', 15),width=10,command=lambda: file_photo())  # 为fm2_left_top添加按钮
#         self.predictButton.pack(side=LEFT)  # 将fm2_left按钮放置在fm2的左边

#         self.truthButton = Button(self.fm2_left_bottom, text='视频', bg='#00CCFF', fg='black',
#                                   font=('微软雅黑', 15), width=10, command=lambda:file_video())
#         self.truthButton.pack(side=LEFT)

#         self.predictButton = Button(self.fm2_left_top, text='转灰度图', bg='#00CCFF', fg='black',
#                                     font=('微软雅黑', 15),width=10, command=lambda:gray_scale_image())  # 为fm2_left_top添加按钮
#         self.predictButton.pack(side=LEFT)  # 将fm2_left按钮放置在fm2的左边

#         self.truthButton = Button(self.fm2_left_bottom, text='转镜像图', bg='#00CCFF', fg='black',
#                                   font=('微软雅黑', 15), width=10, command=lambda:the_mirror())
#         self.truthButton.pack(side=LEFT)

#         self.predictButton = Button(self.fm2_left_top, text='人像美颜', bg='#00CCFF', fg='black',
#                                     font=('微软雅黑', 15), width=10,command=lambda:beautify())  # 为fm2_left_top添加按钮
#         self.predictButton.pack(side=LEFT)  # 将fm2_left按钮放置在fm2的左边

#         self.truthButton = Button(self.fm2_left_bottom, text='转卡通图', bg='#00CCFF', fg='black',
#                                   font=('微软雅黑', 15), width=10,command=lambda:get_cartoon())
#         self.truthButton.pack(side=LEFT)
#         self.fm2_left.pack(side=LEFT)  # fm2_left控件靠左
#         self.nextVideoButton = Button(self.fm2_right_top, text='超配BGM', bg='pink', fg='black',
#                                       font=('微软雅黑', 15), width=10,command=lambda:play_music())#为fm2_right添加按钮
#         self.nextVideoButton.pack(side=RIGHT)#显示按钮，side=LEFT可以不要，下面一句就可以了

#         self.nextVideoButton = Button(self.fm2_right_bottom, text='停止播放', bg='pink', fg='black',
#                                       font=('微软雅黑', 15), width=10, command=lambda: stop_music())  # 为fm2_right添加按钮
#         self.nextVideoButton.pack(side=RIGHT)
#         self.fm2_right_top.pack(side=TOP)
#         self.fm2_right_bottom.pack(side=BOTTOM)
#         self.fm2_right.pack(side=RIGHT)# fm2_right控件靠右
#         self.fm2.pack(side= TOP,expand = 1,fill = 'x')#将fm2放置在顶端，fm1先放置，所以会紧跟fm1下边
#         # fm3
#         self.fm3 = Frame(self)
#         load = Image.open('a2.jpg')#专门打开图片的函数
#         render = ImageTk.PhotoImage(load)#显示图片
#         self.img = Label(self.fm3, image=render)#向fm3中加入组件，指定图片为render
#         self.img.image = render#显示图片
#         self.img.pack()
#         self.fm3.pack(side=TOP)#将fm3放置在顶端，fm1先放置，其次为fm2,最后为fm3

# def take_photos(i):
#     cap = cv2.VideoCapture(0)  # 读取电脑自带摄像头内容
#     while (1):
#         ret, frame = cap.read()  # 按帧读取，返回两个值，ret和frame，ret为布尔值，true为真，文件读取到结尾就返回false，frame是该帧图像的三维矩阵BGR形式。）
#         frame = cv2.flip(frame, 1)
#         k = cv2.waitKey(1)  # 等待键盘输入
#         if k == ord('s'):  # 当键盘输入为s时
#             cv2.imwrite( cwd +'\photos\\' + str(i) + '.jpg', frame)
#             i += 1
#             # 改变number.txt文件中照片的数量
#             file_to_read1 = open('photo_number.txt', 'r+')
#             file_to_read1.seek(0)  # 将指针移动到开头，从第一位开始覆盖写入
#             file_to_read1.write(str(i))
#             file_to_read1.close()
#             j = i - 1
#         if k == ord('v'):
#             img1 = cv2.imread( cwd +'\photos\\' + str(j) + '.jpg')
#             cv2.namedWindow("img1", 0)  # 可调大小
#             cv2.imshow('img1', img1)
#             cv2.waitKey(1)
#         cv2.putText(frame, 'Enter s to take a picture, ESC exit', (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
#         # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
#         #cv2.putText(result, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#         cv2.namedWindow("camera", 0)  # 可调大小
#         cv2.imshow("camera", frame)
#         if k == 27:
#             break
#         # if cv2.getWindowProperty(cap, cv2.WND_PROP_AUTOSIZE) < 2:
#         # break
#     cap.release()#释放内存空间
#     cv2.destroyAllWindows()#删除窗口
#     return i

# #打开相册文件夹
# def file_photo():
#     os.system("start explorer "+cwd+"\photos")

# #录像功能
# def video(k):
#     cap = cv2.VideoCapture(0)
#     # 指定视频编解码方式为XVID
#     codec = cv2.VideoWriter_fourcc(*'XVID')
#     fps = 20.0  # 指定写入帧率为20
#     frameSize = (640, 480)  # 指定窗口大小
#     # # 创建 VideoWriter对象
#     out = cv2.VideoWriter(cwd+'\\videos\\' +str(k) + '.avi', codec, fps, frameSize)

#     k += 1
#     #修改存储的视频数量
#     file_to_read = open('video_number.txt', 'r+')
#     file_to_read.seek(0)  # 将指针移动到开头，从第一位开始覆盖写入
#     file_to_read.write(str(k))
#     file_to_read.close()

#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             frame = cv2.flip(frame, 1)
#             out.write(frame)
#             cv2.imshow("Recording,According to the end of the 'q'", frame)
#             if cv2.waitKey(2) == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# #打开视频文件夹
# def file_video():
#     os.system("start explorer "+cwd+"\\videos")

# #镜像图转换
# def the_mirror():
#     Fpath =filedialog.askopenfilename()
#     img = cv2.imread(Fpath)
#     cv2.imshow("original picture", img)
#     img1 = cv2.flip(img, 1)  # 镜像
#     '''
#     参数2 必选参数。用于指定镜像翻转的类型，其中0表示绕×轴正直翻转，即垂直镜像翻转；1表示绕y轴翻转，即水平镜像翻转；-1表示绕×轴、y轴两个轴翻转，即对角镜像翻转。
#     参数3 可选参数。用于设置输出数组，即镜像翻转后的图像数据，默认为与输入图像数组大小和类型都相同的数组。
#     '''
#     cv2.imshow('change picture', img1)
#     cv2.waitKey(1)

# #灰度图转换
# def gray_scale_image():
#     Fpath = filedialog.askopenfilename()
#     img1 = cv2.imread(Fpath)
#     cv2.namedWindow("original picture", 0)  # 可调大小
#     cv2.imshow('original picture', img1)
#     img2 = cv2.imread(Fpath, 0)
#     #cv2.namedWindow("gray scale image", 0)  # 可调大小
#     cv2.imshow('gray  scale image', img2)
#     cv2.waitKey(1)

# #双边滤镜美化图片
# def beautify():
#     Fpath=filedialog.askopenfilename()
#     img = cv2.imread(Fpath)
#     cv2.imshow('original picture', img)
#     #cv2.bilateralFilter(img,d,’p1’,’p2’)函数有四个参数需要，d是领域的直径，后面两个参数是空间高斯函数标准差和灰度值相似性高斯函数标准差。
#     dst = cv2.bilateralFilter(img, 15, 25, 25)
#     cv2.imshow('change picture', dst)
#     cv2.waitKey(0)

# # 播放BGM
# def play_music():
#     file=cwd+'/BGM2.mp3'
#     # 初始化
#     mixer.init()
#     # 加载音乐文件
#     mixer.music.load(file)
#     # 开始播放音乐流
#     mixer.music.play()

# def stop_music():
#     mixer.music.stop()

# def get_cartoon():
#     request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/selfie_anime"
#     # 二进制方式打开图片文件
#     Fpath = filedialog.askopenfilename()
#     f = open(Fpath, 'rb')
#     img = base64.b64encode(f.read())
#     f.close()
#     host ='自行注册'#！！！！！！！
#     response = requests.get(host)
#     if response:
#         print(response.json()['access_token'])
#     params = {"image":img}
#     access_token = response.json()['access_token']
#     request_url = request_url + "?access_token=" + access_token
#     headers = {'content-type': 'application/x-www-form-urlencoded'}
#     response = requests.post(request_url, data=params, headers=headers)
#     print(response)
#     if response:
#         print (response.json())
#         f = open('cartoon.jpg', 'wb')
#         anime = response.json()['image']#获取动漫头像
#         print(anime)
#         anime = base64.b64decode(anime)#对返回的图像image进行解码
#         f.write(anime)
#         f.close()
#         img = cv2.imread('cartoon.jpg')
#         cv2.namedWindow("cartoon photo", 0)  # 可调大小
#         cv2.imshow("cartoon photo", img)
#         cv2.waitKey(0)

# if __name__=='__main__':
#     cwd = os.getcwd()
#     # 读取number.txt文件夹中的数据，文件夹保存了相册和视频的数量两个数据，读取后赋值给命名序号，然后拍摄后改变数据并覆盖写入number文件夹
#     file_to_read1 = open("photo_number.txt", "r")
#     number1 = file_to_read1.read()
#     list1 = number1.split()
#     i = int(list1[0])
#     file_to_read1.close()
#     print(i)
#     file_to_read2 = open("video_number.txt", "r")
#     number2 = file_to_read2.read()
#     list2 = number2.split()
#     k = int(list2[0])
#     file_to_read2.close()
#     print(k)
#     app = Application()
#     app.__init__()
#     app.window_init()
#     app.createWidgets()
#     app.mainloop()
