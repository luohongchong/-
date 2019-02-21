# -*- coding: UTF-8 -*-
import numpy as np
import requests
from lxml import etree
import re
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import KNN

#以下第一步爬取图片######################################
try:                                                 #
    url='http://cvapi.gdieee.com/get_sudoku/'        #
    r=requests.get(url)#请求超时时间为30秒             #
                                                     #
except:                                              #
    print("产生异常")                                 #
                                                     #
a= random.randint(0,100)                             # 
url2='http://cvapi.gdieee.com/get_sudoku/?id='       #
t=url2+str(a)                                        #
re=requests.get(t)                                   #
fr=open('/home/luo/下载/新3','wb')                    #
fr.write(re.content)                                 #
fr.close()                                           #
#以上第一步爬去图片######################################


img = cv2.imread('/home/luo/下载/新3')
cv2.imwrite('/home/luo/下载/x8.png',img)


###################################################
img = Image.open('/home/luo/下载/x8.png')          #
img_size = img.size                               #
h = img_size[1]  # 图片高度                        #
w = img_size[0]  # 图片宽度                        #
                                                  #
x = 0.022* w#0.022                                #
y = 0.161 * h#0.161                               #
w = 0.964 * w #0.964                              #改变大小！方便接下来分割9*9
h = 0.815 * h#0.815                               #
                                                  #
# 开始截取                                         #
region = img.crop((x, y, x + w, y + h))           #
# 保存图片                                         #
region.save("/home/luo/下载/x8.png")              #
##################################################




## 数独求解算法，回溯法。来源见下面链接##########################网上的方法！！！！！！！！！
## http://stackoverflow.com/questions/1697334/algorithm-for-solving-sudoku
def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *int(i/3), 3 *int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False
#以上为解数读方法####################################################################




def img2vector(filename): 
    """函数将以文本格式出现的32*32的0-1图片，转变成一维特征数组，返回一维数组   
    Keyword argument:
    filename -- 文本格式的图片文件
    """    
    imgvect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()#读行数
        for j in range(32):
            imgvect[0, 32*i + j] = int(linestr[j])
    return imgvect

def handwriteClassfiy(testfile, trainfile, k):
    """函数将trainfile中的文本图片转换成样本特征集和样本类型集，用testfile中的测试样本测试，无返回值   
    Keyword argument:
    testfile -- 测试的32*32
    trainfile -- 样本图片目录
    """
    
    trainFileList = os.listdir(trainfile)
    trainFileSize = len(trainFileList)#返回长度
    labels = []
    trainDataSet = np.zeros((trainFileSize, 1024))
    for i in range(trainFileSize):
        filenameStr = trainFileList[i]
        digitnameStr = filenameStr.split('.')[0]
        digitLabels = digitnameStr.split('_')[0]
        labels.append(digitLabels)
        trainDataSet[i, :] = img2vector(trainfile + '/' + filenameStr)
          
      
    testdigit = testfile
    classifyresult =KNN.classify (testdigit, trainDataSet, labels, k)#！！！！！！！！！！！！！！！！kkk   
    return classifyresult





img = cv2.imread('/home/luo/下载/x8.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## 阈值分割
ret,thresh = cv2.threshold(gray,200,255,1)

## 对二值图像执行膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
dilated = cv2.dilate(thresh,kernel)
#plt.imshow(cv2.cvtColor(dilated,cv2.COLOR_BGR2RGB))

# 轮廓提取，cv2.RETR_TREE表示建立层级结构  image,
contours, hierarchy = cv2.findContours(dilated ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
## 提取小方格，其父轮廓都为0号轮廓

boxes = []

for i in range(len(hierarchy[0])):

    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])#append作用：添加加入

height,width = img.shape[:2]#取长宽！！
box_h = height/9
box_w = width/9
number_boxes = []
## 数独初始化为零阵
soduko = np.zeros((9, 9),np.int32)#全为0的 名为soduko 的矩阵

for j in range(len(boxes)):
    if boxes[j][2] != -1:
        #number_boxes.append(boxes[j])
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])#用最小矩阵包住
        number_boxes.append([x,y,w,h])
        #img = cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
        #img = cv2.drawContours(img, contours, boxes[j][2], (0,255,0), 1)
        
        ## 对提取的数字进行处理
        number_roi = gray[y:y+h, x:x+w]
        ## 统一大小
        resized_roi=cv2.resize(number_roi,(32,32))
        thresh1 = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2) 
        ## 归一化像素值
        normalized_roi = thresh1/255.
        
        ## 展开成一行让knn识别
        sample1 = normalized_roi.reshape((1,1024))
        sample1 = np.array(sample1,np.float32)
        
        filename = '/home/luo/xiazai/trainingDigits/0_0.txt'
        traindir= '/home/luo/xiazai/trainingDigits'
        testdir = sample1
        number=handwriteClassfiy(testdir, traindir,7)#3好像也行，，number这时候为字符
        number=int(number)#转成int型
        ## 识别结果展示
        cv2.putText(img,str(number),(x+w+1,y+h-20), 3, 2., (255, 0, 0), 2, cv2.LINE_AA)
        
        ## 求在矩阵中的位置
        soduko[int(y/box_h)][int(x/box_w)] = number#所以前面必须严格9*9，方便放入soduko矩阵以便解数读
               
        #print(number)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL); 
        cv2.imshow("img", img)
        cv2.waitKey(30)
print("\n生成的数独\n")
print(soduko)
print("\n求解后的数独\n")

## 数独求解
solveSudoku(soduko)
print(soduko)


#验算###############################
print("\n验算：求每行每列的和\n")    #
row_sum = map(sum,soduko)         #
col_sum = map(sum,zip(*soduko))   #
print(list(row_sum))              #
print(list(col_sum))              #
###################################


#print(sum(soduko.transpose))#展示？？？？？？？？/
## 把结果按照位置填入图片中  
for i in range(9):
    for j in range(9):
        x = int((i+0.25)*box_w)
        y = int((j+0.5)*box_h)
        cv2.putText(img,str(soduko[j][i]),(x,y), 3, 2.5, (0, 0, 255), 2, cv2.LINE_AA)
#print(number_boxes)
cv2.namedWindow("img", cv2.WINDOW_NORMAL);
cv2.imshow("img", img)
cv2.waitKey(0)