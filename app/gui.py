import os
from pathlib import Path
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk
import cv2
import torch
import torchvision.transforms as T
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer


class GUI():
    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("遥感船舶检测")
        self.root.geometry()
        if not model_path:
            model_path = Path(os.path.abspath(
                os.path.dirname(__file__))) / 'models'
        elif isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path

        self.opt_frame = tk.Frame(width=20)
        self.opt_frame.grid(row=0, column=0)
        self.mode_label = tk.Label(self.opt_frame, text="成像模式")
        self.mode_label.grid()
        self.mode = tk.IntVar()
        self.mode.set(1)
        self.mode_btn1 = tk.Radiobutton(self.opt_frame,
                                        text="光学",
                                        variable=self.mode,
                                        value=0)
        self.mode_btn2 = tk.Radiobutton(self.opt_frame,
                                        text="Sar",
                                        variable=self.mode,
                                        value=1)
        self.thing_classes = {
            0: ['ship', 'military', 'civil'],
            1: [
                "Wing in ground", "High speed craft", "Passenger", "Cargo",
                "Tanker", "Other"
            ]
        }
        self.mode_btn1.grid()
        self.mode_btn2.grid()
        self.img_btn = tk.Button(self.opt_frame,
                                 text="选择图片",
                                 command=self.select_image)
        self.img_btn.grid()

        self.img_frame = tk.Frame(self.root)
        self.img_frame.grid(row=0, column=1)
        self.img_canvas = tk.Canvas(self.img_frame)
        self.img_canvas.grid(row=0, column=0)

        self.ext_frame = tk.Frame(self.root)
        self.ext_frame.grid(row=1, column=0)
        self.ext_label = tk.Label(self.ext_frame, text="")
        self.ext_label.grid(row=0, column=0)
        self.info_frame = tk.Frame(self.root)
        self.info_frame.grid(row=1, column=1)
        self.info_text = tk.Text(self.info_frame, width=60, height=10)
        self.info_text.grid()

    def run(self):
        self.root.mainloop()

    def select_image(self):
        image_name = askopenfilename(title="选择图片")
        self.img_path = Path(image_name)
        img, preds = self.get_preds(image_name)
        image = ImageTk.PhotoImage(img)
        self.img_canvas.create_image(0, 0, image=image)
        self.img_canvas.grid()
        for pred in preds:
            msg = " ".join([str(int(i)) for i in pred[1:]])
            msg = self.thing_classes[self.mode.get()][
                pred[0]] + " " + msg + '\n'
            self.info_text.insert(tk.END, msg)

    def get_preds(self, image_name):
        if self.mode == 1:
            weights = str(self.model_path / 'optic.pth')
            scale = 1
        else:
            weights = str(self.model_path / 'sar.pth')
            scale = 3
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.05
        cfg.SOLVER.MAX_ITER = 512
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
            self.thing_classes[self.mode.get()])
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        predictor = DefaultPredictor(cfg)
        im = cv2.imread(image_name)
        outputs = predictor(im)
        instances = outputs["instances"].to(torch.device("cpu"))
        v = Visualizer(im[:, :, ::-1], metadata={}, scale=scale)
        v = v.draw_instance_predictions(instances)
        img = T.ToPILImage()(v.get_image()[:, :, ::-1])
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        preds = []
        for i in range(len(boxes)):
            pred = [classes[i], scores[i] * 100, *boxes[i]]
            preds.append(pred)

        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        return img, preds


if __name__ == "__main__":
    gui = GUI()
    gui.run()
