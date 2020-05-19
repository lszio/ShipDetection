import os
import copy
import numpy as np
import cv2
import torch
from pathlib import Path
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import (RotatedCOCOEvaluator, COCOEvaluator,
                                   inference_on_dataset, DatasetEvaluators)
from detectron2.data import (build_detection_train_loader, MetadataCatalog,
                             DatasetCatalog, build_detection_test_loader)
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
        output_dir = cfg.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        evaluators = [
            RotatedCOCOEvaluator(dataset_name, cfg, True, output_dir)
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
        self.predictor = None

    def clear_cache(self):
        output_dir = Path(self.cfg.OUTPUT_DIR)
        if output_dir.exists():
            for f in output_dir.glob("*coco_format*"):
                f.unlink()

    def train(self, resume=True):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = MyTrainer(self.cfg) if self.rotated else DefaultTrainer(
            self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

    def predict(self, im):
        if not self.predictor:
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR,
                                                  "model_final.pth")
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            self.predictor = DefaultPredictor(self.cfg)
        return self.predictor(im)

    def evaluate(self):
        Evaluator = RotatedCOCOEvaluator if self.rotated else COCOEvaluator
        output_dir = self.cfg.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        evaluator = Evaluator(self.datasets['test'][0], self.cfg, False,
                              output_dir)

        val_loader = build_detection_test_loader(self.cfg,
                                                 self.datasets['test'][0])
        model = build_model(self.cfg)
        inference_on_dataset(model, val_loader, evaluator)
        return evaluator.evaluate()

    def test(self, evaluator=None):
        trainer = MyTrainer(self.cfg) if self.rotated else DefaultTrainer(
            self.cfg)
        trainer.resume_or_load(resume=True)
        if not evaluator:
            Evaluator = RotatedCOCOEvaluator if self.rotated else COCOEvaluator
            output_dir = self.cfg.OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            evaluator = Evaluator(self.datasets['test'][0], self.cfg, False,
                                  output_dir)
        return trainer.test(self.cfg, trainer.model, evaluator)

    def draw_preds(self, im):
        outputs = self.predict(im)
        print(outputs)
        visualizerClass = myVisualizer if self.rotated else Visualizer
        v = visualizerClass(im[:, :, ::-1], metadata=self.metadata)
        print(v)
        v = v.draw_instance_predictions(outputs["instances"].to(
            torch.device("cpu")))
        return v.get_image()[:, :, ::-1]

    def inference(self, dataset_name=None, output_dir=None):
        if not dataset_name:
            dataset_name = self.cfg.DATASETS.TEST[0]
        if not output_dir:
            output_dir = self.cfg.OUTPUT_DIR
        dataset = DatasetCatalog.get(dataset_name)
        pred_file = os.path.join(output_dir, dataset_name + "_predictions.txt")
        ann_file = os.path.join(output_dir, dataset_name + "_annotations.txt")
        outputs = []
        for data in dataset:
            image_id = data['image_id']
            im = cv2.imread(data['file_name'])
            preds = self.predict(im)
            instances = preds["instances"].to(torch.device("cpu"))
            boxes = instances.pred_boxes.tensor.numpy()
            boxes = boxes.tolist()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for j in range(len(instances)):
                tmp = [str(image_id), str(classes[j]), str(scores[j])]
                tmp += [str(i) for i in boxes[j]]
                outputs.append(tmp)
        with open(pred_file, 'w') as f:
            for line in outputs:
                f.write(" ".join(line) + "\n")
        with open(ann_file, 'w') as f:
            for data in dataset:
                for instance in data['annotations']:
                    line = [
                        str(data['image_id']),
                        str(instance['category_id']),
                        *[str(i) for i in instance['bbox']]
                    ]
                    f.write(" ".join(line) + "\n")


def get_detector(name):
    if name not in cfgs.keys():
        print("Doesn't have detector {}".formar(name))
    return Detector(name, cfgs[name])


detectors = {
    name: lambda name=name: get_detector(name)
    for name, cfg in cfgs.items()
}

if __name__ == "__main__":
    import pprint
    pprint(detectors)
