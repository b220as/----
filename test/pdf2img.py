# pip install pymupdf

import fitz  # PyMuPDF
from tkinter import Tk, filedialog # ファイル選択ダイアログを開いてPDFファイルを選択する

def pdf_to_images(file_path):
    pdf_document = fitz.open(file_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image = page.get_pixmap()
        image.save(f'data/output_{page_num + 1}.png')

    pdf_document.close()

root = Tk()
root.withdraw()  # ウィンドウを表示しない

file_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])

if file_path:
    try:
        pdf_to_images(file_path)
        print("変換が完了しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
