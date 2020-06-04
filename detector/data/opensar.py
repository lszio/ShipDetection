import os
from pathlib import Path
import math
from PIL import Image
import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def label_transform(label_in, level=1):
    if level == 0:
        return 0
    wrapper = {
        0: "Other",
        1: "Other",
        2: "Other",
        3: "Other",
        4: "Other",
        5: "Other",
        6: "Other",
        7: "Cargo",
        8: "Tanker",
        9: "Other"
    }
    types = ["Cargo", "Tanker", "Other"]
    if type(label_in) == str:
        if label_in not in types:
            label_in = int(label_in) // 10
            if 0 < label_in < 10:
                label_in = wrapper[label_in]
            else:
                label_in = "Other"
        return types.index(label_in)
    else:
        assert label_in > 0
        return types[label_in]


def get_opensar_dicts(path, img_type='all', level=0, part="all", rotated=True):
    targets = []
    folders = [folder for folder in path.glob("S*")]
    assert part in ['all', 'train', 'test']
    num_anns = 0
    num_chips = 0
    num_imgs = 0
    temp = []
    for folder in folders:
        targets = {}
        root = ET.parse(folder / "ship.xml").getroot()
        for ship in root:
            target = {}
            info = ship[0]
            target['ht'] = [int(info[i].text) for i in [1, 2, 3, 4]]
            target['box'] = [int(info[i].text) for i in [7, 8, 5, 6]]
            box = target['box']
            target['center'] = [int(info[9].text), int(info[10].text)]
            target['label'] = ship[1].find("Ship_Type").text
            trainfo = ship[3]
            target['ship_length'] = float(trainfo[0].text)
            target['ship_width'] = float(trainfo[1].text)
            targets['{}_{}'.format(*target['center'])] = target
            num_anns += 1

        img_size_dict = {}
        if not (folder / ".size_cache.txt").exists():
            with open(folder / ".size_cache.txt", 'w') as f:
                for img in (folder / "Patch_Uint8").glob('*'):
                    x, y = img.name.split("_")[-3:-1]
                    name = x + "_" + y
                    size = Image.open(img).size
                    f.write("{} {} {}\n".format(name, size[0], size[1]))
        with open(folder / ".size_cache.txt", 'r') as f:
            for line in f.readlines():
                item = line.strip().split(" ")
                img_size_dict[item[0]] = (item[1], item[2])
        num_chips += len(img_size_dict)
        for image in (folder / "Patch_Uint8").glob('*'):
            x, y, t = image.name.split("_")[-3:]
            if not img_type == 'all':
                if t != img_type:
                    continue
            num_imgs += 1
            target = targets["{}_{}".format(x[1:], y[1:])]
            target['file_name'] = str(image)
            width, height = img_size_dict["{}_{}".format(x, y)]
            target['width'] = int(width)
            target['height'] = int(height)
            temp.append(target)

    dataset_dicts = []
    for index, target in enumerate(temp):
        if math.isnan(target['ship_width']) or math.isnan(
                target['ship_length']):
            continue
        record = {}
        record["file_name"] = target['file_name']
        record["image_id"] = index
        height, width = target['height'], target['width']
        record["height"] = height
        record["width"] = width
        center = target['center']
        ht = target['ht']
        box = target['box']
        cx = (center[0] - box[0]) / (box[2] - box[0]) * width
        cy = (center[1] - box[1]) / (box[3] - box[1]) * height
        w = ((ht[2] - ht[0])**2 + (ht[3] - ht[1])**2)**0.5
        ang = math.asin(abs((ht[3] - ht[1]) / w))
        if (ht[2] - ht[0]) * (ht[3] - ht[1]) > 0:
            ang = -ang
        a = ang / 3.1415926 * 180
        w = w / (box[2] - box[0]) * width
        h = w / target['ship_length'] * target['ship_width'] * 1.2
        if rotated:
            obj = {
                "bbox": [cx, cy, w, h, a],
                "bbox_mode": BoxMode.XYWHA_ABS,
                "category_id": label_transform(target['label'], level=level),
                "iscrowd": 0
            }
        else:
            delta_y = w * math.sin(ang) / 2 * 1.2
            delta_x = w * math.cos(ang) / 2 * 1.2
            t1 = (cx - delta_x, cy - delta_y)
            t2 = (cx + delta_x, cy + delta_y)
            delta_y = h * math.cos(ang) / 2 * 1.2
            delta_x = h * math.sin(ang) / 2 * 1.2
            xs = []
            ys = []
            for t in [t1, t2]:
                xs.append(t[0] - delta_x)
                xs.append(t[0] + delta_x)
                ys.append(t[1] - delta_y)
                ys.append(t[1] + delta_y)
            xmin = min(xs)
            ymin = min(ys)
            xmax = max(xs)
            ymax = max(ys)
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": label_transform(target['label'], level=level),
                "iscrowd": 0
            }
        record["annotations"] = [obj]
        dataset_dicts.append(record)

    categories = {}

    for record in dataset_dicts:
        category_id = record['annotations'][0]['category_id']
        if category_id not in categories.keys():
            categories[category_id] = [record]
        else:
            categories[category_id].append(record)
    dataset_dicts = []
    for category_id, category in categories.items():
        num_ship = len(category)
        dataset_dicts += category[:-num_ship //
                                  5] if part != 'test' else category[
                                      -num_ship // 5:]
    return dataset_dicts


classes_list = {0: ["ship"], 1: ["Cargo", "Tanker", "Other"]}


def register_opensar_datasets():
    for img_type in ['all', 'vh', 'vv']:
        for rotated in [True, False]:
            for part in ['train', 'test']:
                for level, thing_classes in classes_list.items():
                    name = "opensar_" + str(level) + (
                        "_" if img_type == 'all' else "_" + img_type +
                        "_") + ("r_" if rotated else "") + part
                    DatasetCatalog.register(
                        name,
                        lambda level=level, part=part, rotated=rotated,
                        img_type=img_type: get_opensar_dicts(root / "dataset" /
                                                             "OpenSarShip",
                                                             part=part,
                                                             level=level,
                                                             img_type=img_type,
                                                             rotated=rotated))
                    MetadataCatalog.get(name).set(thing_classes=thing_classes)


if __name__ == "__main__":
    register_opensar_datasets()
    DatasetCatalog.list()
    dataset = DatasetCatalog.get("opensar_1_train")
    print(len(dataset))
    cnt = 0
    for i in dataset:
        if not i['width'] == i['height']:
            print(i)
