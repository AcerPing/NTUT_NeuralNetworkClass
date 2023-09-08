import cv2
import numpy as np
import os
folder1 = './IMAGES/ORI'
folder2 = './IMAGES/CAM'

# 提取文件名中的編號
def get_file_num(filename):
    num_str = ''.join(filter(str.isdigit, filename))
    if num_str:
        return int(num_str)
    else:
        return -1  # 返回一個預設值，表示未能提取數字

# 設定膨脹核的大小和形狀
kernel_size = (5, 5)
kernel_shape = cv2.MORPH_ELLIPSE
kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

# 获取兩個文件夾中的所有文件
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

# 按照編號進行排序
files1 = sorted(files1, key=get_file_num)
files2 = sorted(files2, key=get_file_num)

# 逐一比較每個文件
for i in range(len(files1)):
    # 讀取兩張圖像
    img1 = cv2.imread(os.path.join(folder1, files1[i]))
    img2 = cv2.imread(os.path.join(folder2, files2[i]))

    # 轉換為灰度圖
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 計算圖像差異
    diff = cv2.absdiff(gray1, gray2)

    # 膨脹處理
    diff = cv2.dilate(diff, kernel)

    # 二值化處理
    ret, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # 計算白色像素點的比例
    h, w = thresh.shape[:2]
    white_count = cv2.countNonZero(thresh)
    percent = white_count / (h * w)
    if percent > 0.1:
        print('瑕疵')
    else:
        print('無瑕疵')

    # 顯示圖像，等待用戶確認是否正確
    img1 = cv2.resize(img1, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    key = cv2.waitKey(0)
    if key == ord('q'):  # 如果按下 'q' 鍵，則退出程序
        break
cv2.destroyAllWindows()
