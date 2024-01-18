import os

from digit_recognition import DigitRecognizer 
from change_pdf_name import PDFRenamer
# DigitRecognizerクラスのインスタンスを生成
recognizer = DigitRecognizer()


if __name__ == "__main__":
    pdf_renamer = PDFRenamer()
    #pdfを範囲指定して切り取って画像化する機能

    # pngの数字認識をする
    image_path_1 = './data/digits_condition_met.png'
    processed_image_1 = recognizer.process_image(image_path_1)
    predicted_string_1 = recognizer.recognize_digits(processed_image_1) # 学籍番号がpredicted_string_1に入ってる
    print("Predicted string (Image 1):", predicted_string_1)
    # pdfの名前変換
    pdf_renamer.main(new_name=predicted_string_1)

    # 使用例_2の処理
    #image_path_2 = './data/digits_condition_not_met.png'
    #processed_image_2 = recognizer.process_image(image_path_2)
    #predicted_string_2 = recognizer.recognize_digits(processed_image_2)
    #print("Predicted string (Image 2):", predicted_string_2)