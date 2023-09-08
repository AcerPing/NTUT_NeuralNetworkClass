# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:57:02 2023

@author: acer0
"""

import cv2
import os
import numpy as np
import pandas as pd

os.chdir(r"D:/哲平/北科大_碩班_AI學程/碩一課程\314337 類神經網路/期末報告/AOI影像")

# TODO: NG 有瑕疵的電路板
folder1 = './NG/Perfect Defect - NG'
folder2 = './NG/Perfect Defect CAM - NG'

# 讀取兩個文件夾中的文件
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

if files1 != files2: raise Exception("原始資料錯誤")

# # 讀取文件中的編號
# def get_file_num(filename):
#     num_str = ''.join(filter(str.isdigit, filename))
#     if num_str:
#         return int(num_str)
#     else:
#         return -1  #返回一個默認值，表示未能提取數字
    
# # 按照編號進行排序
# files1 = sorted(files1, key=get_file_num)
# files2 = sorted(files2, key=get_file_num)

def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC) # 縮小尺寸
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰階化處理
    ## =========== Total 為像素和初始值等於0 ===========
    Total = 0
    ahash_str = ''
    for i in range(8):                  #累加求像素之和
        for j in range(8):
            Total = Total + gray[i,j]
    avg = Total / 64                     #平均灰度
    ## =========== 灰度大於平均值為1；反之為0 ===========
    for i in range(8):                  
            for j in range(8):
                if  gray[i,j]>avg:
                    ahash_str=ahash_str+'1'
                else:
                    ahash_str=ahash_str+'0'
    return ahash_str

def pHash(image):
    image = cv2.resize(image,(32,32), interpolation=cv2.INTER_CUBIC) # 縮小尺寸
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # 灰階化處理
    # ========== 將灰度圖轉為浮點型，再進行DCT變換 ==========
    dct = cv2.dct(np.float32(image)) # DCT變換
    dct_roi = dct[0:8,0:8] #  縮小DCT：低頻率的訊息會集中於左上角，因此僅取左上角8*8的矩陣
    avreage = np.mean(dct_roi) # 計算像素平均值：計算所有64個畫素的灰度平均值avreage
    # 計算Hash值
    phash = [] 
    for i in range(dct_roi.shape[0]): 
        for j in range(dct_roi.shape[1]): 
            if dct_roi[i,j] > avreage: 
                phash.append(1) 
            else: 
                phash.append(0) 
    # print('PHash => ',phash)
    return phash

## =========== 漢明距離 (Hash值對比對) ===========
def cmpHash(hash1,hash2):              
    n=0                                #n表示漢明距離初始值為0
    if len(hash1)!=len(hash2):         #hash長度不同則返回-1代表傳參出錯
        return -1
    for i in range(len(hash1)):        #for迴圈逐一判斷
        if hash1[i]!=hash2[i]:         #不相等則n+1，最終為相似度
           n=n+1
    return n

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
# compared_files = []

df = pd.DataFrame(columns=['ahash', 'phash', 'dhash', 'label'])

# 逐一比較每個文件
for i in range(len(files1)):
    
# TODO: aHash & pHash
    imgOri = cv2.imread(os.path.join(folder1, files1[i]))
    imgCam = cv2.imread(os.path.join(folder2, files2[i])) # CAM
    
    # aHash比對
    ahashOri = aHash(imgOri)
    ahashCam = aHash(imgCam)
    ahash_diff = cmpHash(ahashOri, ahashCam)
    print(f'aHash比對: {ahash_diff}')
    
    # pHash比對
    phashOri = pHash(imgOri)
    phashCam = pHash(imgCam)
    phash_diff = cmpHash(phashOri, phashCam)
    print(f'pHash比對: {phash_diff}')

# TODO: dHash
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
    dhashOri = dHash(imgOri)
    dhashCam = dHash(imgCam)
    dhash_diff = sum(ch1 != ch2 for ch1, ch2 in zip(dhashOri, dhashCam))

    if dhash_diff > 10:
        # Ori 二值化
        print(dhash_diff)
        ret1, threshOri = cv2.threshold(imgOri, 125, 255, cv2.THRESH_BINARY)

        combined_img = cv2.hconcat([imgOri, threshOri, threshCam])
        cv2.imshow('Combined Images', combined_img)
        cv2.waitKey(0)
        

    # 將差異小於10的圖片加入OK list
    else: 
        print('相似',dhash_diff)
        # compared_files.append(files1[i])
        combined_img = cv2.hconcat([imgOri, imgCam, threshCam])
        cv2.imshow('Combined Images', combined_img)
        cv2.waitKey(0)
    
    df_insert = pd.DataFrame({'ahash': [ahash_diff],
                  'phash': [phash_diff],
                  'dhash': [dhash_diff],
                  'label':['NG']})
    
    df = pd.concat([df, df_insert], ignore_index=True)

cv2.destroyAllWindows()

