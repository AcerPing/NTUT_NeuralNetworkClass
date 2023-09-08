import cv2
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

folder1 = './Images/ORI'
folder2 = './Images/CAM'

# 讀取兩個文件夾中的文件
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

def aHash(grayImg):
    Total = 0
    grayImg = cv2.resize(grayImg, (8, 8))
    ahash_str = ''
    #累加求像素之和
    for i in range(8):                  
        for j in range(8):
            Total = Total + grayImg[i,j]
    avg = Total / 64 
    # 將每個像素與avg比較 大於avg=1 小於avg=0
    for i in range(8):                  
        for j in range(8):
            if  grayImg[i,j]>avg:
                ahash_str=ahash_str+'1'
            else:
                ahash_str=ahash_str+'0'
    return ahash_str

def dHash(grayImg):
    # 差值哈希算法 缩放8*8
    grayImg = cv2.resize(grayImg, (8, 8))
    dhash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(7):
            if grayImg[i, j] > grayImg[i, j+1]: 
                dhash_str = dhash_str+'1'
            else:
                dhash_str = dhash_str+'0'
    return dhash_str


# combined ： imgOri,imgCam,threshOri,threshCam
def showCombinedImg(*args):
    combined_img = cv2.hconcat(args)
    cv2.imshow('Combined Images', combined_img)
    cv2.waitKey(0)


# 回傳 pixeldiff,percent
def pixelDiff_and_Percent(ori, threshCam):
    # 圖像差異 將Ori(可以選擇二值化 或 膨脹) 與 二值化的threshCam 相減的pixel
    pixeldiff = cv2.absdiff(ori, threshCam)
    
    # 計算白點像素比例 
    # shape 會回傳 (rows, columns, channels)。
    # pixeldiff 是灰度圖像，所以只有兩個維度 = 高度（rows = 直 ）和 寬度（columns = 橫 ）。
    h, w = pixeldiff.shape[:2]
    white_count_Ori = cv2.countNonZero(ori)
    white_count_threshCam = cv2.countNonZero(threshCam)
    white_count_pixeldiff = cv2.countNonZero(pixeldiff)

    # '! =' 表示 '不等於' 
    if white_count_Ori != 0 and white_count_threshCam == 0:
        percent = 100

    elif white_count_Ori == 0 and white_count_threshCam == 0:
        percent = 0
    
    elif white_count_Ori == 0 and white_count_threshCam != 0: 
        percent = 100

    else:
        # 白色像素佔比
        # percent = white_count_pixeldiff / (h * w)
        # percent = white_count_Ori / white_count_threshCam
        percent = white_count_pixeldiff / white_count_threshCam

    return pixeldiff,percent

# 抽出藍色點的Cam圖檔資料
def bluePointOut(B):

    ret2, B = cv2.threshold(B, 140, 255, cv2.THRESH_BINARY)
    bluePointCount = cv2.countNonZero(B)

    return bluePointCount


####### Main Function #######

def main(folder1, files1, folder2, files2, clf, df): #, label

    # 建立 比對OK的list
    compared_files = []
    OK_count = 0
    NO_count = 0

    n = 0

    # 逐一讀取每個圖片
    for i in range(len(files1)):
        
        imgOri = cv2.imread(os.path.join(folder1, files1[i]))
        imgCam = cv2.imread(os.path.join(folder2, files2[i])) 

        (B, G, R) = cv2.split(imgCam) # 將imgCam 分3通道 => 抽藍色

        # CAM 轉灰階
        imgCam = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)

        (B1, G1, R1) = cv2.split(imgOri) # 將imgOri 分3通道 => 抽紅色

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

        # 二值化 => threshOri threshCam
        ret1, threshOri = cv2.threshold(imgOri, 125, 255, cv2.THRESH_BINARY)
        ret2, threshCam = cv2.threshold(imgCam, 140, 255, cv2.THRESH_BINARY)

        # Otsu 二值化 => oriOtsu camOtsu
        ret1, oriOtsu = cv2.threshold(imgOri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, camOtsu = cv2.threshold(imgCam, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ori膨脹 dilate 一次 (iterations = 1) => oriDilated
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        oriDilated = cv2.dilate(threshOri, kernel, iterations = 1)
            # get(獲取)Structuring(結構化) Element(元素) 
            # => 矩形：MORPH_RECT;交叉形：MORPH_CROSS;橢圓形：MORPH_ELLIPSE

      ## aHash比對 imgOri對比imgCam => 40 > ahashdiff閥值 > 4 
        ahashOri = aHash(imgOri)
        ahashCam = aHash(imgCam)
        ahashdiff = sum(ch1 != ch2 for ch1, ch2 in zip(ahashOri, ahashCam))

      ## dHash比對 imgOri對比threshCam => 閥值 >= 39
        dhashOri = dHash(imgOri)
        dhashCam = dHash(threshCam)
        dhashdiff = sum(ch1 != ch2 for ch1, ch2 in zip(dhashOri, dhashCam))

      ## PIXEL 相減比較法          
        # 用二值 threshOri & threshCam 直接相減 產生 pixeldiff
        pixeldiff, percent = pixelDiff_and_Percent(threshOri, threshCam)

      ## ahash Otsu 二值比對 oriOtsu 對比 threshCam
        ahashOriOtsu = aHash(oriOtsu)
        ahashCamThres = aHash(threshCam)
        ahashdiffOtsu = sum(ch1 != ch2 for ch1, ch2 in zip(ahashOriOtsu, ahashCamThres))


    ### 直接使用MLP ###
      
      ## 抽出藍點NG
        bluePointCount = bluePointOut(B)

        print(f'MLP辨識圖像名稱:{files1[i]}')
        
        features = pd.DataFrame({'ahashdiff': [ahashdiff],
                                'dhashdiff': [dhashdiff],
                                'percent': [percent],
                                'ahashdiffOtsu': [ahashdiffOtsu]})
        predict = clf.predict(features)
        
        df_insert = pd.DataFrame({'File Name': [files1[i]],
                                'ahashdiff': [ahashdiff],
                                'dhashdiff': [dhashdiff],
                                'percent': [percent],
                                'ahashdiffOtsu': [ahashdiffOtsu],
                                'Prediction':clf.predict(features)}) #'label':[label]
        
        df = pd.concat([df, df_insert], ignore_index=True)
        
        # print('ahashdiff =', ahashdiff, ' dhashdiff =', dhashdiff, ' percent =', percent , ' ahashdiffOtsu =', ahashdiffOtsu)
        print(f'predict:{predict}') #label:{label}, 

    OK_count += df['Prediction'].value_counts()['OK']
    NO_count += df['Prediction'].value_counts()['NG']
    print('OK_count', OK_count)
    print('NO_count', NO_count)
    cv2.destroyAllWindows()
    
    if not df.empty:
        return df

if __name__ == "__main__":
    
    os.chdir(r"D:/哲平/北科大_碩班_AI學程/碩一課程/314337 類神經網路/機器視覺/Final_Projet")
    df = pd.DataFrame(columns=['File Name', 'ahashdiff', 'dhashdiff', 'percent', 'ahashdiffOtsu', 'Prediction']) #'label'
    
    # TODO: NG 有瑕疵的電路板
    folder1 = './Images(OK/ORI'
    folder2 = './Images(OK/CAM'
    
    # 讀取兩個文件夾中的文件
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    
    if files1 != files2: raise Exception("原始資料錯誤")
    
    filename = 'MLP_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    df = main(folder1, files1, folder2, files2,clf, df) #label='NG'
    
    # print(f"總共{len(df)}個,錯了{len(df[df['label'] !=  df ['Prediction']])}個")
    
    # if not df.empty:
    #     mat = pd.DataFrame(confusion_matrix(df['label'],  df['Prediction']), index=['Label_NG', 'Label_OK'], columns=['Predict_NG', 'Predict_OK'])
        
    #     print('NG')
    #     precision = (mat['Predict_NG']['Label_NG']/sum(mat['Predict_NG']))*100
    #     print('Precision精確率', round(precision,2), '%')
    #     recall = (mat['Predict_NG']['Label_NG']/sum(mat.iloc[0]))*100
    #     print('Recall', round(recall,2), '%')
    #     F1 = 2 * (precision * recall) / (precision + recall)
    #     print('F1-Score', round(F1,2), '%')
        
    #     print('OK')
    #     precision = (mat['Predict_OK']['Label_OK']/sum(mat['Predict_OK']))*100
    #     print('Precision精確率', round(precision,2), '%')
    #     recall = (mat['Predict_OK']['Label_OK']/sum(mat.iloc[1]))*100
    #     print('Recall', round(recall,2), '%')
    #     F1 = 2 * (precision * recall) / (precision + recall)
    #     print('F1-Score', round(F1,2), '%')
        
    #     mat = pd.concat([mat, pd.DataFrame({'Predict_NG':[sum(mat['Predict_NG'])], 'Predict_OK':[sum(mat['Predict_OK'])]}, index=['Sum'])])
    #     mat['Sum'] = pd.Series([sum(mat.iloc[0]), sum(mat.iloc[1]), sum(mat.iloc[2])], index=['Label_NG', 'Label_OK', 'Sum'])
    #     mat

        
        # df.to_csv('Testing.csv', index=False)
    

    







