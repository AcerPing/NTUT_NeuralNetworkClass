import cv2
import os

folder1 = './Images/ORI'
folder2 = './Images/CAM'

# 讀取文件中的編號
def get_file_num(filename):
    num_str = ''.join(filter(str.isdigit, filename))
    if num_str:
        return int(num_str)
    else:
        return -1  #返回一個默認值，表示未能提取數字

# 讀取兩個文件夾中的文件
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

# 按照編號進行排序
files1 = sorted(files1, key=get_file_num)
files2 = sorted(files2, key=get_file_num)

score = 0

# 創建固定大小的視窗
cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Diff', cv2.WINDOW_NORMAL)

# 逐一比較每個文件
for i in range(len(files1)):
    img1 = cv2.imread(os.path.join(folder1, files1[i]))
    img2 = cv2.imread(os.path.join(folder2, files2[i]),0) # CAM

    (B1, G1, R1) = cv2.split(img1)
    cv2.imshow('R1',R1)


    # # 中值濾波器
    # img1 = cv2.medianBlur(R1, 3)
    # img2 = cv2.medianBlur(img2, 3)

    # ret1, thresh1 = cv2.threshold(img1, 100, 255, 0)
    # ret2, thresh2 = cv2.threshold(img2, 100, 255, 0)


    # img1 = cv2.resize(thresh1, (400, 400))
    # img2 = cv2.resize(thresh2, (400, 400))
    # cv2.imshow('img1',img1)
    # cv2.imshow('img2',img2)
    
    # # 圖像差異
    # diff = cv2.absdiff(img1, img2)

    # cv2.imshow('diff1',diff)

    # 計算白點像素比例
    # h, w = diff.shape[:2]
    # white_count = cv2.countNonZero(diff)
    # percent = white_count / (h * w)

    # if percent > 0.1:
    #     print('瑕疵')

    #     # 二值化
    #     ret1, thresh1 = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY)
    #     ret2, thresh2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)

    #     # 圖像差異
    #     diff = cv2.absdiff(thresh1, thresh2)

    #     # 計算白點像素比例
    #     h, w = diff.shape[:2]
    #     white_count = cv2.countNonZero(diff)
    #     percent = white_count / (h * w)

    #     if percent > 0.1:
    #         print('瑕疵')

    #         # 侵蝕
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    #         dilated1 = cv2.dilate(thresh1, kernel, iterations=1)

    #         # 圖像差異
    #         diff = cv2.absdiff(dilated1, thresh2)

    #         # 計算白點像素比例
    #         h, w = diff.shape[:2]
    #         white_count = cv2.countNonZero(diff)
    #         percent = white_count / (h * w)

    #     else:
    #         print('無瑕疵')
    #         score += 0.5
    # else:
    #     print('無瑕疵')
    #     score += 0.5


    # # 調整圖片大小
    # img_1 = cv2.resize(img1, (400, 400))
    # img_2 = cv2.resize(img2, (400, 400))
    # dilated1 = cv2.resize(dilated1, (400, 400))
    # diff = cv2.resize(diff, (400, 400))

    # print('分數:', score)
    # cv2.imshow('img1', img_1)
    # cv2.imshow('img2', img_2)
    # cv2.imshow('Diff', diff)
    # cv2.imshow('Processed', dilated1)

    cv2.waitKey(0)
