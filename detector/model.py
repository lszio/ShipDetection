import os
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from .data import register_datasets

root = Path(os.path.abspath(os.path.dirname(__file__))).parent


def get_model_config(name="hrsc", level=0, rotated=False, pure=False):
    config_name = name + "_" + str(level) + ("_p" if pure else
                                             "") + ("_r" if rotated else "")
    output_dir = root / "outputs" / config_name
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = str(output_dir)
    if name == "hrsc":
        train_datasets = (config_name + "_train", config_name + "_val")
        test_datasets = (config_name + "_test", )
    else:
        train_datasets = (config_name + "_train", )
        test_datasets = (config_name + "_test", )
    if (output_dir / "model_final.pth").exists():
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        if not rotated:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = test_datasets

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.05
    cfg.SOLVER.MAX_ITER = 6300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        MetadataCatalog.get(train_datasets[0]).thing_classes)
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    if rotated:
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, -60, -30, 0, 30, 60]]
        cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 10.0)

    return cfg


register_datasets()
cfgs = {
    "hrsc_0": get_model_config(),
    "hrsc_0_r": get_model_config(rotated=True),
    "hrsc_1": get_model_config(level=1),
    "hrsc_1_r": get_model_config(rotated=True, level=1),
    "hrsc_2": get_model_config(level=2),
    "hrsc_2_r": get_model_config(rotated=True, level=2),
    "hrsc_1_p": get_model_config(level=1, pure=True),
    "hrsc_1_p_r": get_model_config(rotated=True, level=1, pure=True),
    "hrsc_2_p": get_model_config(level=2, pure=True),
    "hrsc_2_p_r": get_model_config(rotated=True, level=2, pure=True),
    "opensar_0": get_model_config('opensar', rotated=False),
    "opensar_1": get_model_config('opensar', level=1, rotated=False),
    "opensar_0_r": get_model_config('opensar', rotated=True),
    "opensar_1_r": get_model_config('opensar', level=1, rotated=True)
}

if __name__ == "__main__":
    import pprint
    pprint.pprint(cfgs)
