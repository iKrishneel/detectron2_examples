#!/usr/bin/env python

from dataclasses import dataclass
from typing import Collection

from torch.optim import Optimizer

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.data.build import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler

from detectron2_examples.faster_rcnn import Dataloader, DatasetMapper
from detectron2_examples.faster_rcnn.car196 import Car196


def register_dataset(dataset_root: str, dataset_name: str):
    DatasetCatalog.register(
        dataset_name, Car196(root=dataset_root)
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes = ["car", ],
        thing_color = [(0, 255, 0), ],
    )
    

def register_datasets(
        dataset_roots: Collection[str], dataset_names: Collection[str]
):
    for dataset_root, dataset_name in zip(
            dataset_roots, dataset_names
    ):
        register_dataset(dataset_root, dataset_name)


def read_all_dataset_dicts(dataset_names: Collection[str]):
    assert len(dataset_names)

    dataset_name_to_dicts = {}
    for dataset_name in dataset_names:
        dataset_name_to_dicts[dataset_name] = DatasetCatalog.get(dataset_name)



class Trainer(DefaultTrainer):

    root: str = None
    dataloader = None
    
    def __init__(self, cfg):
        # super(Trainer, self).__init__(cfg=cfg)

        # data_source = Car196(root=cfg.DATASETS.TRAIN)
        # self.dataloader = Dataloader(data_source=data_source)

        print("Init")
        
    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):
        mapper = DatasetMapper(cfg, True) if mapper is None else mapper
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer: Optimizer):
        return build_d2_lr_scheduler(cfg, optimizer)


def add_faster_rcnn_config(cfg):
    cfg.INPUT.ROTATION_ANGLES = 10


def setup(args):
    cfg = get_cfg()
    add_faster_rcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=True, type=str)
    args = parser.parse_args()
    print(args)

    cfg = setup(args)

    register_datasets([args.root, ], cfg.DATASETS.TRAIN)
    read_all_dataset_dicts(cfg.DATASETS.TRAIN)

    t = Trainer(cfg)
    import IPython
    IPython.embed()
