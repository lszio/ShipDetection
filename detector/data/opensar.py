import os
from pathlib import Path
import math
from PIL import Image
# import cv2
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
        2: "Wing in ground",
        3: "Other",
        4: "High speed craft",
        5: "Other",
        6: "Passenger",
        7: "Cargo",
        8: "Tanker",
        9: "Other"
    }
    types = [
        "Wing in ground", "High speed craft", "Passenger", "Cargo", "Tanker",
        "Other"
    ]
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


def get_opensar_dicts(path, img_type='vv', level=0, part="all", rotated=True):
    imgs = {'vv': [], 'vh': []}
    targets = []
    num_images = 0
    folders = [folder for folder in path.glob("S*")]
    sorted(folders)
    assert part in ['all', 'train', 'test']
    for folder in folders:
        img_dict = {}
        img_size_dict = {}
        if not (folder / "chip_size.txt").exists():
            for folder in path.iterdir():
                with open(folder / "chip_size.txt", 'w') as f:
                    for img in (folder / "Patch_Uint8").iterdir():
                        x, y = img.name.split("_")[-3:-1]
                        name = x + "_" + y
                        size = Image.open(img).size
                        f.write("{} {}\n".format(name, size[0]))
        for folder in path.iterdir():
            with open(folder / "chip_size.txt", 'r') as f:
                for line in f.readlines():
                    item = line.strip().split(" ")
                    img_size_dict[item[0]] = item[1]
        for image in (folder / "Patch_Uint8").iterdir():
            if image.name[0] == '.':
                continue
            x, y, t = image.name.split("_")[-3:]
            img_dict["{}_{}_{}".format(x[1:], y[1:],
                                       t[:2])] = (image,
                                                  img_size_dict[x + "_" + y])
            num_images += 1
        root = ET.parse(folder / "ship.xml").getroot()
        for ship in root:
            target = {}
            info = ship[0]
            target['ht'] = [int(info[i].text) for i in [1, 2, 3, 4]]
            target['box'] = [int(info[i].text) for i in [7, 8, 5, 6]]
            target['center'] = [int(info[9].text), int(info[10].text)]
            target['label'] = ship[1].find("Ship_Type").text
            trainfo = ship[3]
            target['length'] = float(trainfo[0].text)
            target['width'] = float(trainfo[1].text)
            targets.append(target)
            imgs['vv'].append(img_dict["{}_{}_vv".format(*target['center'])])
            imgs['vh'].append(img_dict["{}_{}_vh".format(*target['center'])])

    dataset_dicts = []
    for index, img_info in enumerate(imgs[img_type]):
        record = {}
        img, size = img_info
        filename = str(img)

        record["file_name"] = filename
        record["image_id"] = index
        height = width = int(size)
        record["height"] = height
        record["width"] = width
        cx, cy = width / 2, height / 2
        ht = targets[index]['ht']
        box = targets[index]['box']
        w = ((ht[2] - ht[0])**2 + (ht[3] - ht[1])**2)**0.5
        ang = math.asin(abs(ht[3] - ht[1]) / w)
        a = ang / 3.1415926 * 180
        if (ht[2] - ht[0]) * (ht[3] - ht[1]) > 0:
            a = -a
        w = w / (box[2] - box[0]) * width
        h = w / targets[index]['length'] * targets[index]['width']
        if rotated:
            obj = {
                "bbox": [cx, cy, w, h, a],
                "bbox_mode":
                BoxMode.XYWHA_ABS,
                "category_id":
                label_transform(targets[index]['label'], level=level),
                "iscrowd":
                0
            }
        else:
            hl = (w**2 + h**2)**0.5 * 0.5
            x = hl * math.cos(ang) * 1.5
            y = hl * math.sin(ang) * 2.1
            obj = {
                "bbox": [cx - x, cy - y, cx + x, cy + y],
                "bbox_mode":
                BoxMode.XYXY_ABS,
                "category_id":
                label_transform(targets[index]['label'], level=level),
                "iscrowd":
                0
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
        print(num_ship)
        dataset_dicts += category[:-num_ship //
                                  5] if part != 'test' else category[
                                      -num_ship // 5:]
    return dataset_dicts


classes_list = {
    0: ["ship"],
    1: [
        "Wing in ground", "High speed craft", "Passenger", "Cargo", "Tanker",
        "Other"
    ]
}


def register_opensar_datasets():
    for rotated in [True, False]:
        for part in ['train', 'test']:
            for level, thing_classes in classes_list.items():
                name = "opensar_" + str(level) + ("_r" if rotated else
                                                  "") + "_" + part
                DatasetCatalog.register(
                    name,
                    lambda level=level, part=part, rotated=rotated:
                    get_opensar_dicts(root / "dataset" / "OpenSarShip",
                                      part=part,
                                      level=level,
                                      rotated=rotated))
                MetadataCatalog.get(name).set(thing_classes=thing_classes)


if __name__ == "__main__":
    register_opensar_datasets()
    DatasetCatalog.list()
    dataset = DatasetCatalog.get("opensar_1_train")
    print(dataset[0])
