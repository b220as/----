import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import fitz

def canvas_to_pdf_coordinates(canvas_x, canvas_y, pdf_resolution=72):
    pdf_x = canvas_x * pdf_resolution / 300  # 300dpiを基準に調整
    pdf_y = canvas_y * pdf_resolution / 300  # 300dpiを基準に調整
    return pdf_x, pdf_y

def pdf_to_image(pdf_path, page_number, rect, image_path):
    pdf_document = fitz.open(pdf_path)
    
    # ページ取得
    page = pdf_document[page_number - 1]
    
    # 元のサイズで画像を取得
    pix = page.get_pixmap()
    
    # 切り取る範囲を指定
    rect = fitz.Rect(rect)
    
    # 切り取り
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 切り取った範囲を PIL Image に変換
    cropped_img = img.crop((rect.x0, rect.y0, rect.x1, rect.y1))
    
    # 画像を保存
    cropped_img.save(image_path)

    pdf_document.close()

class PDFCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Cutter")

        self.pdf_path = ""
        self.page_number = 1
        self.rect_start = None
        self.rect_end = None

        # GUI要素の配置
        self.label = tk.Label(root, text="Select PDF file:")
        self.label.pack()

        self.pdf_button = tk.Button(root, text="Browse PDF", command=self.load_pdf)
        self.pdf_button.pack()

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_click_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_click_end)

        self.cut_button = tk.Button(root, text="Cut and Save Image", command=self.cut_and_save)

    def load_pdf(self):
        self.pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if self.pdf_path:
            self.display_page()
            self.cut_button.pack()

    def display_page(self):
        pdf_document = fitz.open(self.pdf_path)
        page = pdf_document[self.page_number - 1]
        
        # 元のサイズで画像を取得
        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        pdf_document.close()

    def on_click_start(self, event):
        # Canvas上の座標をPDFページ上の座標に変換
        pdf_x, pdf_y = canvas_to_pdf_coordinates(event.x, event.y)
        print(f"Start Click: Canvas({event.x}, {event.y}) PDF({pdf_x}, {pdf_y})")
        self.rect_start = (pdf_x, pdf_y)

    def on_drag(self, event):
        if self.rect_start:
            # Canvas上の座標をPDFページ上の座標に変換
            pdf_x, pdf_y = canvas_to_pdf_coordinates(event.x, event.y)
            print(f"Dragging: Canvas({event.x}, {event.y}) PDF({pdf_x}, {pdf_y})")
            self.canvas.delete("rect")
            self.canvas.create_rectangle(
                self.rect_start[0], self.rect_start[1], pdf_x, pdf_y,
                outline="red", width=2, tags="rect"
            )

    def on_click_end(self, event):
        # Canvas上の座標をPDFページ上の座標に変換
        pdf_x, pdf_y = canvas_to_pdf_coordinates(event.x, event.y)
        print(f"End Click: Canvas({event.x}, {event.y}) PDF({pdf_x}, {pdf_y})")
        self.rect_end = (pdf_x, pdf_y)

    def cut_and_save(self):
        if self.pdf_path and self.rect_start and self.rect_end:
            left = min(self.rect_start[0], self.rect_end[0])
            top = min(self.rect_start[1], self.rect_end[1])
            right = max(self.rect_start[0], self.rect_end[0])
            bottom = max(self.rect_start[1], self.rect_end[1])

            rect = (left, top, right, bottom)
            image_path = "data/image.png"

            pdf_to_image(self.pdf_path, self.page_number, rect, image_path)
            print(f"Image saved to {image_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFCutterApp(root)
    root.mainloop()


