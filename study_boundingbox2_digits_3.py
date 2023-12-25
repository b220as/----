#
# 手書き連番数字の認識
#
#

from pathlib import Path
import cv2
from sklearn import svm
import pickle
# import numpy as np

# 入力画面サイズに応じて調整．... (入力画像の面積)*0.5, etc. とすべき ...
THRESH_MIN_AREA = 10
THRESH_MAX_AREA = 500000

def main():
    # 学習済モデルの読み込み
    parent = Path(__file__).resolve().parent
    file_name = str(parent.joinpath("./data/trained_model_svc_28x28_g1e-3.pkl"))
    clf = pickle.load(open(file_name, 'rb'))

    img = cv2.imread('./data/digits.png') # ファイル指定別の形に

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # グレースケール化
    img = cv2.bitwise_not(img) # 白黒反転（ネガ）
    _thre, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # 大津の2値化
    cv2.imwrite('./data/digits_mask.jpeg', img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        if len(contours[i]) > 0: #右の数字から認識される

            # 認識範囲外のものを除去
            area1 = cv2.contourArea(contours[i])
            if (area1 < THRESH_MIN_AREA) or (area1 > THRESH_MAX_AREA):  
                continue

            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)

            # 切り出し
            cropped = img[y:y+h, x:x+w] #この時余白を付けるといいのかもしれない

            # 画像を28x28にリサイズ
            resized = cv2.resize(cropped, (28, 28)) # サイズ変更 => 28 * 28

            # 識別の実行（SVCを使って予測）
            predicted = clf.predict(resized.reshape(1, -1))

            # 予測結果の表示(確認用)
            print("Prediction:", predicted)

            #ここで画像を保存する(確認用)数字は認識されている
            #filename = f'resized_digit_{i}.jpg'
            #cv2.imwrite(filename, resized)
        
if __name__ == '__main__':
    main()



# グレースケール化、ネガ、リサイズ、2値化の順番によって結果が変わるかもしれない