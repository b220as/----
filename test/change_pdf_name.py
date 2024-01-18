# pdf_renamer.py
import os
import fitz  # PyMuPDF
from tkinter import filedialog, Tk

class PDFRenamer:
    def __init__(self):
        self.root = Tk()
        self.root.withdraw()

    def _check_file_exists(self, file_path):
        if not os.path.exists(file_path):
            print(f"指定されたファイルは存在しません: {file_path}")
            return False
        return True

    def _check_file_extension(self, file_path):
        if not file_path.lower().endswith(".pdf"):
            print("指定されたファイルはPDFではありません。")
            return False
        return True

    def rename_pdf(self, file_path, new_name):
        if not self._check_file_exists(file_path) or not self._check_file_extension(file_path):
            return

        base_path, ext = os.path.splitext(file_path)
        new_file_name = f"{new_name}{ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        os.rename(file_path, new_file_path)
        print(f"ファイル名を変更しました: {file_path} → {new_file_path}")

    def main(self, new_name):
        file_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])

        if file_path:
            try:
                self.rename_pdf(file_path, new_name)
            except Exception as e:
                print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    pdf_renamer = PDFRenamer()
    pdf_renamer.main("your_predefined_string")
