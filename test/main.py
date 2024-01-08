# Strategy Patternを使用して、数字認識の戦略を切り替え可能にするためのクラスとファクトリーメソッドを実装しました。
# SimpleConvNetRecognitionクラスはDigitRecognitionStrategyを実装し、数字認識アルゴリズムをカプセル化しています。
# DigitRecognizerFactoryは異なる認識アルゴリズムのインスタンス化を抽象化し、新しい認識アルゴリズムの追加を容易にします。

import pickle
import cv2
import numpy as np
from abc import ABC, abstractmethod

from simple_convnet import SimpleConvNet

class DigitRecognitionStrategy(ABC):
    @abstractmethod
    def recognize(self, img):
        pass

class SimpleConvNetRecognition(DigitRecognitionStrategy):
    def __init__(self, thresh_min_area, thresh_max_area):
        self.network = SimpleConvNet(input_dim=(1, 28, 28), 
                                     conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                                     hidden_size=100, output_size=10, weight_init_std=0.01)
        self.network.load_params()
        self.THRESH_MIN_AREA = thresh_min_area
        self.THRESH_MAX_AREA = thresh_max_area

    def recognize(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  

        predictions_list = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.THRESH_MIN_AREA or area > self.THRESH_MAX_AREA:
                continue
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
            predictions = self.network.predict(resized)
            predicted_class = np.argmax(predictions)
            predictions_list.append(str(predicted_class))  # 予測結果を文字列としてリストに追加

            # 画像の保存
            resized = resized.reshape(28, 28)
            filename = f'./output/test_{predicted_class}_{i}.jpg'
            cv2.imwrite(filename, resized)

        predicted_string = ''.join(predictions_list)
        return predicted_string

class DigitRecognizerFactory:
    @staticmethod
    def create_recognizer(strategy,thresh_min_area, thresh_max_area):
        if strategy == "SimpleConvNet":
            return SimpleConvNetRecognition(thresh_min_area, thresh_max_area)
        # 他の数字認識アルゴリズムがあればここで追加

class DigitRecognizer:
    def __init__(self, strategy="SimpleConvNet",thresh_min_area=10, thresh_max_area=500000):
        self.recognition_strategy = DigitRecognizerFactory.create_recognizer(strategy,thresh_min_area, thresh_max_area)
        self.THRESH_MIN_AREA = 10
        self.THRESH_MAX_AREA = 500000

    def process_image(self, image_path):
        # 画像の読み込みと前処理
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        return img

    def recognize_digits(self, img):
        # 数字の認識処理
        predicted_string = self.recognition_strategy.recognize(img)
        return predicted_string

# 使用例
recognizer = DigitRecognizer()
image_path = './data/digits0.png'
processed_image = recognizer.process_image(image_path)
predicted_string = recognizer.recognize_digits(processed_image)
print("Predicted string:", predicted_string)
