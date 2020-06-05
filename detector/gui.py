import os
from pathlib import Path
import tkinter as tk
import time
import threading
from tkinter import filedialog, simpledialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from .engine import detectors

ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent


class GUI():
    def __init__(self):
        self.imgs = []
        self.img_id = 0
        self.root = tk.Tk()
        self.root.title("遥感船舶检测")
        # self.root.geometry("600x400+200+100")
        self.info = tk.StringVar()
        self.modes = []
        self.mode_id = tk.IntVar()
        self.mode_id.set(0)
        self.menubar = self.init_menu()
        self.detector = None
        self.init_weights()
        self.init_app()
        self.change_mode()

    def init_weights(self):
        img_frame = tk.Frame(self.root)
        img_frame.grid(row=0, column=1)

        info_label = tk.Label(img_frame,
                              justify=tk.LEFT,
                              textvariable=self.info)
        info_label.grid(row=0, column=0, columnspan=3)
        info = "图片位置：{}\n图片大小：{} x {}\n"
        self.info.set(info.format(" ", "0", "0"))

        image_label = tk.Label(img_frame)
        image_label.grid(row=1, column=0, columnspan=3)
        self.image_label = image_label
        inference_btn = tk.Button(img_frame,
                                  text="标注",
                                  command=self.do_inference)
        inference_btn.grid(row=2, column=0)
        next_btn = tk.Button(img_frame,
                             text="下一张",
                             command=lambda x=1: self.next_image(x))
        last_btn = tk.Button(img_frame,
                             text="上一张",
                             command=lambda x=-1: self.next_image(x))
        last_btn.grid(row=2, column=1)
        next_btn.grid(row=2, column=2)

        log_text = ScrolledText(img_frame, height=10)
        log_text.grid(row=3, column=0, columnspan=3)
        self.log_text = log_text

    def init_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开文件", command=self.open_file)
        file_menu.add_command(label="打开文件夹", command=self.open_folder)
        file_menu.add_command(label="保存", command=self.save)
        file_menu.add_command(label="保存全部", command=self.save_all)
        menubar.add_cascade(label="文件", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="分割", command=self.do_split)
        menubar.add_cascade(label="操作", menu=edit_menu)

        mode_menu = tk.Menu(menubar, tearoff=0)
        model_path = ROOT / 'models'
        index = 0
        for path in model_path.iterdir():
            if not (path / 'model_final.pth').exists():
                continue
            mode_menu.add_radiobutton(label=path.name,
                                      value=index,
                                      variable=self.mode_id,
                                      command=self.change_mode)
            self.modes.append(path.name)
            index += 1
        menubar.add_cascade(label='模式', menu=mode_menu)
        menubar.add_command(label='关于')
        self.root.config(menu=menubar)
        return menubar

    def init_app(self):
        self.open_folder('.')

    def run(self):
        self.root.mainloop()

    def open_file(self):
        path = filedialog.askopenfilenames()
        if not path:
            return
        self.imgs = list(path)
        self.show_image()

    def change_mode(self):
        name = self.modes[self.mode_id.get()]
        self.detector = detectors[name]()
        self.log("切换到{}模式".format(name), clear=True)

    def open_folder(self, path=None):
        if not path:
            path = filedialog.askdirectory()
            if not path:
                return
        path = Path(path)
        self.imgs = []
        for file_path in path.iterdir():
            name = file_path.name
            img_types = ['tif', 'jpg', 'png', 'bmp']
            if not name.split('.')[-1] in img_types or name[0] == '.':
                continue
            self.imgs.append(str(file_path))
        self.log('检测到{}张图片'.format(len(self.imgs)))
        self.show_image()

    def show_image(self, img=None):
        if not img:
            img = self.get_image()
            if not img:
                return
        scale = img.size[0] / 256
        self.info.set("图片位置：{}\n图片大小：{} x {}\n第{}张/共{}张\n".format(
            self.imgs[self.img_id], img.size[0], img.size[1], self.img_id + 1,
            len(self.imgs)))
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def inference(self, img=None):
        start = time.time()
        self.log("正在标注", end='...')
        # detector = detectors[self.modes[self.mode_id.get()]]()
        if not img:
            img = self.get_image()
        img_p, result = self.detector.inference_pil(img)
        self.show_image(img_p)
        end = time.time()
        self.log("标注完成， 共耗时{:.2f}s".format(end - start))
        self.log("共检测到{}艘船".format(len(result)))
        for item in result:
            bbox = "{: <4.0f},{: <4.0f},{: <4.0f},{: <4.0f}".format(
                *item['bbox'])
            if len(item['bbox']) > 4:
                bbox += ",{: >2.0f}\u00B0".format(item['bbox'][-1])
            self.log(" 类别：{: <9}, 置信度：{: >3.0%}, 位置：{}".format(
                item['class'], item['score'], bbox))
        return img_p, result

    def do_inference(self):
        thread = threading.Thread(target=self.inference)
        thread.start()
        return thread

    def save(self):
        target = filedialog.asksaveasfilename()
        img_p, result = self.inference()
        img_p.save(target, quality=95)
        self.log("保存完成")

    def save_all(self):
        target = filedialog.askdirectory()
        if target:
            target = Path(target)
        else:
            return
        for index in range(len(self.imgs)):
            img = self.get_image(index)
            filename = str(target / (str(index) + '.jpg'))
            img_p, result = self.inference(img)
            img_p.save(filename)
        self.log("全部保存完成")

    def next_image(self, step=1):
        self.img_id += step
        self.show_image()

    def split(self, size=256, padding=16):
        img = self.get_image()
        width, height = img.size
        step = size - padding
        x, y = width // step + 1, height // step + 1
        boxes = []
        self.log("开始分割", end="...")
        for i in range(x):
            for j in range(y):
                x1, x2 = i * step, i * step + 256
                y1, y2 = j * step, j * step + 256
                if width - x1 < size / 2 or height - y1 < size / 2:
                    continue
                if x2 > width:
                    x2 = width
                    x1 = x2 - size
                if y2 > height:
                    y2 = height
                    y1 = y2 - size
                boxes.append([x1, y1, x2, y2])
        self.log("原始大小：{}x{}，目标大小：{}x{}，共{}张".format(width, height, size, size,
                                                     len(boxes)))
        imgs = [img.crop(box) for box in boxes]
        self.imgs = imgs
        return imgs

    def do_split(self):
        size, padding = MyDialog(self.root).result
        self.split(size, padding)
        self.show_image()

    def get_image(self, index=None):
        num_img = len(self.imgs)
        if num_img == 0:
            return
        if not index:
            if not 0 <= self.img_id < num_img:
                self.img_id += num_img if self.img_id < 0 else -num_img
            index = self.img_id
        img = self.imgs[index]
        if isinstance(img, str):
            img = Image.open(img)
        return img

    def log(self, msg, end='\n', clear=False):
        if clear:
            self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, msg + end)


class MyDialog(simpledialog.Dialog):
    def body(self, master, title=None):

        size_label = tk.Label(master, text="大小")
        size_entry = tk.Entry(master)
        padding_label = tk.Label(master, text="边距")
        padding_entry = tk.Entry(master)

        size_label.grid(row=0, column=0)
        size_entry.grid(row=0, column=1)
        padding_label.grid(row=1, column=0)
        padding_entry.grid(row=1, column=1)
        size_entry.insert(0, 256)
        padding_entry.insert(0, 16)
        self.size_entry = size_entry
        self.padding_entry = padding_entry

    def apply(self):
        size = int(self.size_entry.get())
        padding = int(self.padding_entry.get())

        self.result = size, padding
        return size, padding


if __name__ == "__main__":
    gui = GUI()
    gui.run()
