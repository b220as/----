import pickle
import cv2
import numpy as np
from simple_convnet import SimpleConvNet

# 保存されたパラメータの読み込み
with open('params.pkl', 'rb') as f:
    params = pickle.load(f)

# SimpleConvNetのインスタンスを生成
network = SimpleConvNet(input_dim=(1, 28, 28), 
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
# 学習済みの重みをセット
network.load_params()

# 画像の前処理パラメータ
THRESH_MIN_AREA = 10
THRESH_MAX_AREA = 500000

# 画像の読み込み
img = cv2.imread('./data/digits0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

# 数字の輪郭を取得しX軸の位置でソート
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  

predictions_list = []

# 各数字の処理
for i, contour in enumerate(contours):
    # 輪郭の面積を計算し、範囲外のものを除去
    area = cv2.contourArea(contour)
    if area < THRESH_MIN_AREA or area > THRESH_MAX_AREA:
        continue
    
    # 数字を切り出し
    x, y, w, h = cv2.boundingRect(contour)
    cropped = img[y:y+h, x:x+w]

    # 追加する余白のサイズ
    max_side_length = max(cropped.shape[0], cropped.shape[1])
    margin = int(0.2 * max_side_length)
    
    # 余白を追加して画像を28x28にリサイズ
    cropped_with_margin = cv2.copyMakeBorder(cropped, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(cropped_with_margin, (28, 28))
    resized = resized.reshape(1, 1, 28, 28)

    # 数字認識の実行
    predictions = network.predict(resized)
    predicted_class = np.argmax(predictions)
    predictions_list.append(str(predicted_class))  # 予測結果を文字列としてリストに追加

    # 画像の保存
    resized = resized.reshape(28, 28)
    filename = f'./output/test_{predicted_class}_{i}.jpg'
    cv2.imwrite(filename, resized)

# リスト内の数字を文字列として結合
predicted_string = ''.join(predictions_list)
print("Predicted string:", predicted_string)
