import os
import json
import cv2
import torch
from collections import OrderedDict
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt


class MyEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self.dataset_name = dataset_name
        self._distributed = distributed
        if not output_dir:
            output_dir = cfg.OUTPUT_DIR
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        self._num_classses = len(self._metadata.thing_classes)
        self._result = OrderedDict()
        self._predictions = []
        self._annotations = []

    def reset(self):
        self._result = OrderedDict()
        self._predictions = {i: [] for i in range(self._num_classses)}
        self._annotations = {i: [] for i in range(self._num_classses)}

    def process(self, inputs, outputs):
        for i, o in zip(inputs, outputs):
            instances = o["instances"].to(self._cpu_device)
            num_instance = len(instances)
            if num_instance == 0:
                continue
            boxes = instances.pred_boxes.tensor.numpy()
            if boxes.shape[1] == 4:
                boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS,
                                        BoxMode.XYWH_ABS)
            boxes = boxes.tolist()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for k in range(num_instance):
                result = {
                    "image_id": i['image_id'],
                    "bbox": boxes[k],
                    "score": scores[k]
                }
                self._predictions[classes[k]].append(result)

    def evaluate(self):
        self._get_annotations()
        self._store()
        self._result['Count'] = self._count()
        anns = []
        for category_id, category in self._annotations.items():
            for record in category:
                ann = [record['image_id'], category_id, *record['bbox']]
                anns.append(ann)
        preds = []
        for category_id, category in self._predictions.items():
            for record in category:
                pred = [
                    record['image_id'], category_id, record['score'],
                    *record['bbox']
                ]
                preds.append(pred)
        calc = DetectionEval(anns, preds)
        result = calc.inference()
        self._result["AP"] = {
            self._metadata.thing_classes[k]: v['ap']
            for k, v in result.items()
        }
        print(self._result)
        return self._result

    def _get_annotations(self):
        self._annotations = {i: [] for i in range(self._num_classses)}
        dataset = DatasetCatalog.get(self.dataset_name)
        for data in dataset:
            for instance in data['annotations']:
                category = instance['category_id']
                result = {
                    'image_id': data['image_id'],
                    'bbox': instance['bbox']
                }
                self._annotations[category].append(result)

    def _count(self):
        result = {
            self._metadata.thing_classes[k]: len(self._predictions[k])
            for k in range(self._num_classses)
        }
        return result

    def _store(self):
        with open(os.path.join(self._output_dir, "predictions.json"),
                  'w') as f:
            json.dump(self._predictions, f, indent=4)
        with open(os.path.join(self._output_dir, "annotations.json"),
                  'w') as f:
            json.dump(self._annotations, f, indent=4)


class DetectionEval():
    def __init__(self, anns=None, preds=None):
        self._predictions = {}
        self._annotations = {}
        if anns and preds:
            if isinstance(anns, list):
                self.update_dict(*self.get_data_from_list(anns, preds))

    def get_list_from_txt(self, annFile, predFile):
        with open(annFile, 'r') as f:
            anns = [line.split(" ") for line in f.readlines()]
        with open(predFile, 'r') as f:
            preds = [line.split(" ") for line in f.readlines()]
        return (anns, preds)

    def get_data_from_list(self, anns, preds):
        annotations = {}
        predictions = {}
        for ann in anns:
            category = int(ann[1])
            img_id = ann[0]
            bbox = [float(i) for i in ann[2:]]
            if category not in annotations.keys():
                annotations[category] = {}
            if img_id not in annotations[category].keys():
                annotations[category][img_id] = []
            annotations[category][img_id].append({'bbox': bbox})
        for pred in preds:
            category = int(pred[1])
            img_id = pred[0]
            score = float(pred[2])
            bbox = [float(i) for i in pred[3:]]
            if category not in predictions.keys():
                predictions[category] = {}
            if img_id not in predictions[category].keys():
                predictions[category][img_id] = []
            predictions[category][img_id].append({
                'bbox': bbox,
                'score': score,
                'tp': False
            })

        return (annotations, predictions)

    def update_dict(self, annotations, predictions):
        self._annotations = annotations
        self._predictions = predictions

    def inference(self, threshold=0.5):
        ann_count = {
            category_id: 0
            for category_id in self._annotations.keys()
        }
        for category_id, category in self._annotations.items():
            for image_id, instances in category.items():
                ann_count[category_id] += len(instances)
                for instance in instances:
                    if category_id not in self._predictions.keys():
                        break
                    if image_id not in self._predictions[category_id].keys():
                        continue
                    preds = self._predictions[category_id][image_id]
                    ious = []
                    for pred in preds:
                        ious.append(self.iou(instance['bbox'], pred['bbox']))
                    max_iou = max(ious)
                    if max_iou > threshold:
                        index = ious.index(max_iou)
                        self._predictions[category_id][image_id][index][
                            'tp'] = True
        result = {}
        for category_id, category in self._predictions.items():
            preds = []
            for image_id, instances in category.items():
                for instance in instances:
                    preds.append((instance['score'], instance['tp']))
            preds = sorted(preds, key=lambda a: -a[0])

            ps = []
            rs = []
            acc_ap = 0
            acc_ar = 0
            for pred in preds:
                if pred[1]:
                    acc_ap += 1
                else:
                    acc_ar += 1
                ps.append(acc_ap / (acc_ap + acc_ar))
                rs.append(acc_ap / ann_count[category_id])
            result[category_id] = {
                'ap': self.my_ap(ps, rs),
                'precision': ps[-1],
                'recall': rs[-1]
            }
            self.draw_curve(ps, rs)
        return result

    @classmethod
    def iou(cls, box1, box2):
        if len(box1) == len(box2) == 4:
            in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
            in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
            inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                    (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
            return inter / union
        elif len(box1) == len(box2) == 5:
            assert len(box1) == len(box2) == 5
            a = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            b = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            i = cv2.rotatedRectangleIntersection(a, b)
            if i[1] is None:
                return 0
            return cv2.contourArea(i[1])

    @classmethod
    def draw_curve(cls, ps, rs):
        plt.title("Precision x Recall curve")
        plt.xlabel('Recall')
        plt.xlabel('Precision')
        plt.plot(rs, ps)
        plt.show()

    @classmethod
    def my_ap(cls, ps, rs):
        pre = 0
        ap = 0
        for p, r in zip(ps, rs):
            ap += p * (r - pre)
            pre = r
        return ap
