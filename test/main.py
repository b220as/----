from pathlib import Path
import cv2
import numpy as np
from simple_convnet import SimpleConvNet
import pickle

# 保存されたパラメータの読み込み
with open('params.pkl', 'rb') as f:
    params = pickle.load(f)

# SimpleConvNetのインスタンスを生成
network = SimpleConvNet(input_dim=(1, 28, 28), 
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
# 学習済みの重みをセット
network.load_params()

# 入力画面サイズに応じて調整．... (入力画像の面積)*0.5, etc. とすべき ...
THRESH_MIN_AREA = 10
THRESH_MAX_AREA = 500000
MARGIN = 10  # 追加する余白のサイズ

# 画像の前処理
img = cv2.imread('./data/digits1.png') # ファイル指定別の形に
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # グレースケール化
img = cv2.bitwise_not(img) # 白黒反転（ネガ）
_thre, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # 大津の2値化
#cv2.imwrite('./data/digits_mask.jpeg', img)
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

predictions_list = []

# 画像の認識

# X軸の位置でソート
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  

for i, contour in enumerate(contours):
    if len(contours[i]) > 0: 
        # 認識範囲外のものを除去
        area1 = cv2.contourArea(contours[i])
        if (area1 < THRESH_MIN_AREA) or (area1 > THRESH_MAX_AREA):  
            continue

        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        # ここでOpenCVを使用して新しいデータを用意し、適切に前処理を行う

        # 切り出し
        cropped = img[y:y+h, x:x+w] 

        # 縦と横の長い方の長さを取得
        max_side_length = max(cropped.shape[0], cropped.shape[1])

        # 追加する余白のサイズ（縦あるいは横の長い方の2割）
        MARGIN = int(0.2 * max_side_length)

        # 余白を追加
        cropped_with_margin = cv2.copyMakeBorder(cropped, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=0)

        # 画像を28x28にリサイズ
        resized = cv2.resize(cropped_with_margin, (28, 28)) # サイズ変更 => 28 * 28
        resized = resized.reshape(1, 1, 28, 28) # この変形は間違いか

        # 数字認識の実行
        predictions = network.predict(resized)
        predicted_class = np.argmax(predictions)

        # 予測結果の表示(確認用)
        #print("Prediction:", predicted_class)

        # 予測結果を文字列としてリストに追加
        predictions_list.append(str(predicted_class))  

        #ここで画像を保存する(確認用)数字は認識されている
        resized = resized.reshape(28, 28)
        filename = f'./output/test_{predicted_class}_{i}.jpg'
        cv2.imwrite(filename, resized)

# リスト内の数字を文字列として結合
predicted_string = ''.join(predictions_list)
print("Predicted string:", predicted_string)