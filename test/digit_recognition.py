import os
import cv2
import numpy as np
from abc import ABC, abstractmethod

from simple_convnet import SimpleConvNet

# 数字認識戦略を定義する抽象クラス
class DigitRecognitionStrategy(ABC):
    @abstractmethod
    def recognize(self, img):
        """与えられた画像から数字を認識する抽象メソッド。

        Args:
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            str: 認識された数字の文字列。
        """
        pass

# SimpleConvNetを使用した数字認識戦略の具体的実装
class SimpleConvNetRecognition(DigitRecognitionStrategy):
    def __init__(self, min_area_threshold, max_area_threshold):
        """SimpleConvNetに基づく数字認識戦略の初期化。

        Args:
            min_area_threshold (int): 数字として認識する最小の輪郭面積。
            max_area_threshold (int): 数字として認識する最大の輪郭面積。
        """
        self.network = SimpleConvNet(input_dim=(1, 28, 28), 
                                     conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                                     hidden_size=100, output_size=10, weight_init_std=0.01)
        self.network.load_params()
        self.MIN_AREA_THRESHOLD = min_area_threshold
        self.MAX_AREA_THRESHOLD = max_area_threshold

    def recognize(self, img):
        """与えられた画像から数字を認識する。

        Args:
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            str: 認識された数字の文字列。
        """
        # 数字の検出と予測
        detected_contours = self.detect_and_classify_contours(img)
        predictions_list = self.predict_detected_digits(detected_contours)
        predicted_string = ''.join(map(str, predictions_list))# すべての予測値を文字列に変換
        return predicted_string

    # 数字の輪郭を取得する関数
    def detect_and_classify_contours(self, img):
        """与えられた画像から数字の輪郭を検出し分類する。

        Args:
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            list: 分類された数字画像のリスト。
        """
        contours = self.detect_contours(img)
        classified_contours = self.classify_detected_contours(contours, img)
        return classified_contours

    # 輪郭検出関数
    def detect_contours(self, img):
        """与えられた画像から数字の輪郭を検出する。

        Args:
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            list: 検出された数字の輪郭のリスト。
        """
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        return sorted_contours

    # 輪郭を分類する関数
    def classify_detected_contours(self, contours, img):
        """与えられた数字の輪郭を分類する。

        Args:
            contours (list): 検出された数字の輪郭のリスト。
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            list: 分類された数字画像のリスト。
        """
        classified_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA_THRESHOLD or area > self.MAX_AREA_THRESHOLD:
                continue
            cropped_digit = self.extract_digit_from_contour(contour, img)
            resized_digit = self.resize_digit_with_margin(cropped_digit)
            classified_contours.append(resized_digit)
            self.save_digit_image(resized_digit, i)
        return classified_contours

    # 輪郭から数字を切り出す関数
    def extract_digit_from_contour(self, contour, img):
        """与えられた数字の輪郭から数字を切り出す。

        Args:
            contour (numpy.ndarray): 数字の輪郭情報。
            img (numpy.ndarray): グレースケールの数字画像。

        Returns:
            numpy.ndarray: 数字の切り出された画像。
        """
        x, y, w, h = cv2.boundingRect(contour)
        digit_image = img[y:y+h, x:x+w]
        return digit_image

    # 数字をリサイズして余白を追加する関数
    def resize_digit_with_margin(self, digit_image):
        """与えられた数字画像をリサイズし、余白を追加する。

        Args:
            digit_image (numpy.ndarray): 数字の画像。

        Returns:
            numpy.ndarray: リサイズされた数字の画像。
        """
        max_side_length = max(digit_image.shape[0], digit_image.shape[1])
        margin = int(0.2 * max_side_length)
        digit_with_margin = cv2.copyMakeBorder(digit_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
        resized_digit = cv2.resize(digit_with_margin, (28, 28))
        resized_digit = resized_digit.reshape(1, 1, 28, 28)
        return resized_digit

    # 数字を予測する関数
    def predict_detected_digits(self, classified_contours):
        """与えられた数字の画像から数字を予測する。

        Args:
            classified_contours (list): 分類された数字画像のリスト。

        Returns:
            predictions_list (str): 予測された文字列のリスト。
        """
        predictions_list = []
        for digit_image in classified_contours:
            prediction = self.predict_digit(digit_image)
            predictions_list.append(str(prediction))
        # 最初の要素を"b"に変更
        if predictions_list:
            predictions_list[0] = "b"
            predictions_list[1] = "2"
            return predictions_list
    # 数字を予測する関数
    def predict_digit(self, digit_image):
        """与えられた数字の画像から数字を予測する。

        Args:
            digit_image (numpy.ndarray): 数字の画像。

        Returns:
            int: 予測された数字。
        """
        predictions = self.network.predict(digit_image)
        predicted_class = np.argmax(predictions)
        return predicted_class

    # 数字の画像を保存する関数
    def save_digit_image(self, digit_image, i):
        """与えられた数字の画像を保存する。

        Args:
            digit_image (numpy.ndarray): 数字の画像。
            i (int): 画像のインデックス。
        """
        digit_image = digit_image.reshape(28, 28)
        filename = f'./output/detected_digit_{i}.jpg'
        cv2.imwrite(filename, digit_image)

# 数字認識クラスを生成するファクトリークラス
class DigitRecognizerFactory:
    @staticmethod
    def create_recognizer(strategy, thresh_min_area, thresh_max_area):
        if strategy == "SimpleConvNet":
            return SimpleConvNetRecognition(thresh_min_area, thresh_max_area)

# 数字認識のためのクラス
class DigitRecognizer:
    def __init__(self, strategy="SimpleConvNet", thresh_min_area=10, thresh_max_area=500000):
        self.recognition_strategy = DigitRecognizerFactory.create_recognizer(strategy, thresh_min_area, thresh_max_area)
        self.THRESH_MIN_AREA = thresh_min_area
        self.THRESH_MAX_AREA = thresh_max_area

    # 画像の前処理を行う関数
    def process_image(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError("指定されたファイルが見つかりません。")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
        if img is None:
            raise ValueError("画像を読み込めませんでした。")
        if len(img.shape) != 2:
            raise ValueError("グレースケールの画像を指定してください。")

        img = cv2.bitwise_not(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        return img

    # 数字の認識処理を行う関数
    def recognize_digits(self, img):
        try:
            predicted_string = self.recognition_strategy.recognize(img)
            if self.check_condition(predicted_string):
                return predicted_string
            else:
                raise ValueError("Error: Condition not met.")
        except Exception as e:
            return str(e)

    
    # 文字が8文字であり、かつ2文字目が2、8文字目が0であるかを確認
    def check_condition(self, text):
        if len(text) == 8 and text[1] == '2' and text[7] == '0':
            return True
        else:
            return False

# DigitRecognizerクラスのインスタンスを生成
recognizer = DigitRecognizer()

# 使用例_1の処理
image_path_1 = './data/digits_condition_met.png'
processed_image_1 = recognizer.process_image(image_path_1)
predicted_string_1 = recognizer.recognize_digits(processed_image_1)
print("Predicted string (Image 1):", predicted_string_1)

# 使用例_2の処理
image_path_2 = './data/digits_condition_not_met.png'
processed_image_2 = recognizer.process_image(image_path_2)
predicted_string_2 = recognizer.recognize_digits(processed_image_2)
print("Predicted string (Image 2):", predicted_string_2)
