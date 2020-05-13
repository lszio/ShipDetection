import os
import copy
import numpy as np
import torch
from pathlib import Path
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import (RotatedCOCOEvaluator, DatasetEvaluators,
                                   COCOEvaluator, inference_on_dataset)
from detectron2.data import (build_detection_train_loader, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from .model import cfgs


def my_transform_instance_annotations(annotation,
                                      transforms,
                                      image_size,
                                      *,
                                      keypoint_hflip_indices=None):
    annotation["bbox"] = transforms.apply_rotated_box(
        np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))
    annos = [
        my_transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        evaluators = [
            RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)
        ]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)


class myVisualizer(Visualizer):
    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            masks = None
            keypts = None
            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS)
                for x in annos
            ]
            labels = [x["category_id"] for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels,
                                   boxes=boxes,
                                   masks=masks,
                                   keypoints=keypts)
        return self.output


class Detector():
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.datasets = {
            "train": self.cfg.DATASETS.TRAIN,
            "test": self.cfg.DATASETS.TEST
        }
        self.rotated = (name[-1] == 'r')
        self.metadata = MetadataCatalog.get(self.datasets['train'][0])
        self.clear_cache()

    def clear_cache(self):
        inference_dir = Path(self.cfg.OUTPUT_DIR) / "inference"
        if inference_dir.exists():
            for f in inference_dir.glob("*coco_format*"):
                f.unlink()

    def train(self, resume=False):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = MyTrainer(self.cfg) if self.rotated else DefaultTrainer(
            self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

    def predict(self, im):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR,
                                              "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(self.cfg)
        return predictor(im)

    def eval(self):
        Evaluator = RotatedCOCOEvaluator if self.rotated else COCOEvaluator
        output_dir = os.path.join(self.cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_dir, exist_ok=True)
        evaluator = Evaluator(self.datasets['test'][0], self.cfg, False,
                              output_dir)

        val_loader = build_detection_test_loader(self.cfg,
                                                 self.datasets['test'][0])
        model = build_model(self.cfg)
        inference_on_dataset(model, val_loader, evaluator)
        return evaluator.evaluate()


detectors = {name: Detector(name, cfg) for name, cfg in cfgs.items()}

if __name__ == "__main__":
    import pprint
    pprint(detectors)
