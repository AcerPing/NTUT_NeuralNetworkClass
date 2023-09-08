# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:57:02 2023

@author: acer0
"""

import cv2
import os
import numpy as np

folder1 = './Images/ORI'
folder2 = './Images/CAM'

# 讀取兩個文件夾中的文件
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

# 讀取文件中的編號
def get_file_num(filename):
    num_str = ''.join(filter(str.isdigit, filename))
    if num_str:
        return int(num_str)
    else:
        return -1  #返回一個默認值，表示未能提取數字
    
# # 按照編號進行排序
# files1 = sorted(files1, key=get_file_num)
# files2 = sorted(files2, key=get_file_num)

def dHash(img):
    # 差值哈希算法 缩放8*8
    img = cv2.resize(img, (8, 8))
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(7):
            if img[i, j] > img[i, j+1]: 
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

# 建立 比對OK的list
compared_files = []

# 逐一比較每個文件
for i in range(len(files1)):
    imgOri = cv2.imread(os.path.join(folder1, files1[i]))
    imgCam = cv2.imread(os.path.join(folder2, files2[i]), 0) # CAM + 灰階

    (B1, G1, R1) = cv2.split(imgOri) # 將imgOri 分3通道

### 影像前處理 ###

    # 中值濾波器 5x5
    imgOri = cv2.medianBlur(R1, 5) # 將imgOri 取R通道
    imgCam = cv2.medianBlur(imgCam, 5)

    # 高斯濾波
    imgOri = cv2.GaussianBlur(imgOri, (5, 5), 0)
    imgCam = cv2.GaussianBlur(imgCam, (5, 5), 0)

    # 中值濾波器 3x3
    imgOri = cv2.medianBlur(imgOri, 3) 
    imgCam = cv2.medianBlur(imgCam, 3)

    # 邊緣強化 Ori
    sigma = 25
    blur_img = cv2.GaussianBlur(imgOri, (0, 0), sigma)
        # 以原圖 : 模糊圖片= 1.5 : -0.5 的比例進行混合
    imgOri = cv2.addWeighted(imgOri, 1.5, blur_img, -0.5, 0)


    # Cam 二值化
    ret2, threshCam = cv2.threshold(imgCam, 140, 255, cv2.THRESH_BINARY)

    # dHash比對
    hashOri = dHash(imgOri)
    hashCam = dHash(imgCam)
    diff = sum(ch1 != ch2 for ch1, ch2 in zip(hashOri, hashCam))

    if diff > 10:
        # Ori 二值化
        print(diff)
        ret1, threshOri = cv2.threshold(imgOri, 125, 255, cv2.THRESH_BINARY)


        combined_img = cv2.hconcat([imgOri, threshOri, threshCam])
        cv2.imshow('Combined Images', combined_img)
        cv2.waitKey(0)
        

    # 將差異小於10的圖片加入OK list
    else: 
        print('相似',diff)
        compared_files.append(files1[i])
        combined_img = cv2.hconcat([imgOri, imgCam, threshCam])
        cv2.imshow('Combined Images', combined_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()