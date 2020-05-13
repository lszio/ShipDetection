import os
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def hrsc_label_transform(label_in, level=1):
    label_dict = {
        '100000001': ['船', '船', '船', '船', '船'],
        '100000002': ['船', '军用', '航母', '航母', '航母'],
        '100000003': ['船', '军用', '军舰', '军舰', '军舰'],
        '100000004': ['船', '民用', '商船', '商船', '商船'],
        '100000005': ['船', '军用', '航母', '航母', '尼米兹级航母'],
        '100000006': ['船', '军用', '航母', '航母', '企业级航母'],
        '100000007': ['船', '军用', '军舰', '驱逐舰', '阿利伯克级驱逐舰'],
        '100000008': ['船', '军用', '军舰', '登陆舰', '惠德贝岛级船坞登陆舰'],
        '100000009': ['船', '军用', '军舰', '护卫舰', '佩里级护卫舰'],
        '100000010': ['船', '军用', '军舰', '运输舰', '圣安东尼奥级两栖船坞运输舰'],
        '100000011': ['船', '军用', '军舰', '巡洋舰', '提康德罗加级巡洋舰'],
        '100000012': ['船', '军用', '航母', '航母', '小鹰级航母'],
        '100000013': ['船', '军用', '航母', '航母', '俄罗斯库兹涅佐夫号航母'],
        '100000014': ['船', '军用', '军舰', '护卫舰', '阿武隈级护卫舰'],
        '100000015': ['船', '军用', '军舰', '运输舰', '奥斯汀级两栖船坞运输舰'],
        '100000016': ['船', '军用', '军舰', '攻击舰', '塔拉瓦级通用两栖攻击舰'],
        '100000017': ['船', '军用', '军舰', '指挥舰', '蓝岭级指挥舰'],
        '100000018': ['船', '民用', '商船', '货船', '集装箱货船'],
        '100000019': ['船', '军用', '军舰', '指挥舰', '尾部OX头部圆指挥舰'],
        '100000020': ['船', '民用', '商船', '运输汽车船', '运输汽车船'],
        '100000022': ['船', '民用', '商船', '气垫船', '气垫船'],
        '100000024': ['船', '民用', '商船', '游艇', '游艇'],
        '100000025': ['船', '民用', '商船', '货船', '货船'],
        '100000026': ['船', '民用', '商船', '游轮', '游轮'],
        '100000027': ['船', '军用', '潜艇', '潜艇', '潜艇'],
        '100000028': ['船', '军用', '军舰', '军舰', '琵琶形军舰'],
        '100000029': ['船', '军用', '军舰', '医疗船', '医疗船'],
        '100000030': ['船', '民用', '商船', '运输汽车船', '运输汽车船'],
        '100000031': ['船', '军用', '航母', '航母', '福特级航空母舰'],
        '100000032': ['船', '军用', '航母', '航母', '中途号航母'],
        '100000033': ['船', '军用', '航母', '航母', '无敌级航空母舰']
    }
    types = {k: [] for k in range(4)}
    for k in types.keys():
        for i in label_dict.keys():
            if (label_dict[i][k] not in types[k]):
                types[k].append(label_dict[i][k])

    if label_in == "metadata":
        return types[level]
    else:
        if type(label_in) == str:
            if label_in in label_dict.keys():
                label_in = label_dict[label_in][level]
            return types[level].index(label_in)
        else:
            assert label_in > 0
            return types[level][label_in]


def get_hrsc_dicts(path, part='train', level=1, rotated=True):
    with open(path / "ImageSets" / "{}.txt".format(part), 'r') as f:
        img_ids = f.readlines()
    img_ids = sorted([id.strip() for id in img_ids])
    dataset_dicts = []
    for img_id in img_ids:
        record = {}
        filename = os.path.join(path / "AllImages" / "{}.bmp".format(img_id))
        record["file_name"] = filename
        record["image_id"] = int(img_id)

        root = ET.parse(path / "Annotations" /
                        "{}.xml".format(img_id)).getroot()
        record["location"] = [
            float(i) for i in root.find("Img_Location").text.split((','))
        ]
        record["height"] = int(root.find("Img_SizeHeight").text)
        record["width"] = int(root.find("Img_SizeWidth").text)
        objs = []
        for ship in root.find("HRSC_Objects"):
            label = ship.find("Class_ID").text
            if rotated:
                bbox = [float(ship[i].text) for i in [9, 10, 11, 12, 13]]
            else:
                bbox = [float(ship[i].text) for i in [5, 6, 7, 8]]
            obj = {
                "bbox": bbox,
                "bbox_mode":
                BoxMode.XYWHA_ABS if rotated else BoxMode.XYXY_ABS,
                "category_id": hrsc_label_transform(label, level),
                "iscrowd": 0
            }
            obj['bbox'][-1] /= math.pi / 180 / -1
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


classes_list = {
    1: ["ship"],
    2: ["ship", "military", 'civil'],
    3: ["ship", "aircraft carrier", "warship", "submarine", "merchant"]
}


def register_hrsc_datasets():
    for rotated in [True, False]:
        for level, thing_classes in classes_list.items():
            for d in ["train", "val", "test"]:
                DatasetCatalog.register(
                    "hrsc_" + str(level) + ("_r_" if rotated else "_") + d,
                    lambda d=d: get_hrsc_dicts(root / "dataset" / "HRSC",
                                               d,
                                               level=level,
                                               rotated=rotated))
                MetadataCatalog.get("hrsc_" + str(level) +
                                    ("_r_" if rotated else "_") +
                                    d).set(thing_classes=thing_classes)


if __name__ == "__main__":
    DatasetCatalog.clear()
    register_hrsc_datasets()
    print(DatasetCatalog.list())
