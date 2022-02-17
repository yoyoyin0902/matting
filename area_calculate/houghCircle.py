import  cv2


img=cv2.imread('1931136.jpg')
cv2.imshow('img',img)
#灰度化
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#輸出影象大小，方便根據影象大小調節minRadius和maxRadius
print(img.shape)
#霍夫變換圓檢測
circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
#輸出返回值，方便檢視型別
print(circles)
#輸出檢測到圓的個數
print(len(circles[0]))

print('-------------我是條分割線-----------------')
#根據檢測到圓的資訊，畫出每一個圓
for circle in circles[0]:
    #圓的基本資訊
    print(circle[2])
    #座標行列
    x=int(circle[0])
    y=int(circle[1])
    #半徑
    r=int(circle[2])
    #在原圖用指定顏色標記出圓的位置
    img=cv2.circle(img,(x,y),r,(0,0,255),-1)
#顯示新影象
cv2.imshow('res',img)

#按任意鍵退出
cv2.waitKey(0)
cv2.destroyAllWindows()
