#!/usr/bin/env python

from dataclasses import dataclass
import argparse
from typing import Collection

from torch.optim import Optimizer

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.data import (
    # DatasetMapper,
    detection_utils, DatasetCatalog, MetadataCatalog
)
from detectron2.data.build import build_detection_train_loader
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler

from detectron2_examples.mask_rcnn import PennFudanDataset, DatasetMapper


def register_dataset(
        dataset_root: str, dataset_name: str, is_train: bool = True
):
    DatasetCatalog.register(
        dataset_name, PennFudanDataset(
            root=dataset_root,
            is_train=is_train
        )
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=[
            'person'
        ],
        thing_color=[
            (0, 255, 0)
        ]
    )

    
def register_datasets(
        dataset_roots: Collection[str], dataset_names: Collection[str],
        is_train: bool = True
):
    for dataset_root, dataset_name in zip(dataset_roots, dataset_names):
        register_dataset(
            dataset_root, dataset_name, is_train=is_train
        )


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):
        augmentations = cls.build_augmentation(cfg, True)
        if mapper is None:
            mapper = DatasetMapper(
                cfg=cfg,
                is_train=True,
                # augmentations=augmentations,
                # use_instance_mask=True,
                # recompute_boxes=True
            )

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_schedule(cls, cfg: CfgNode, optim: Optimizer):
        return build_d2_lr_schedule(cfg, optimizer)

    @classmethod
    def build_augmentation(cls, cfg: CfgNode, is_train: bool = True):
        result = detection_utils.build_augmentation(
            cfg, is_train=is_train
        )
        return result

    
def add_mask_rcnn_config(cfg, args):
    cfg.OUTPUT_DIR = args.output_dir

    try:
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError:
        pass


def setup(args) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_datasets([args.root,], cfg.DATASETS.TRAIN)
    register_datasets([args.root,], cfg.DATASETS.TEST, is_train=False)
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
    

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    args = parser.parse_args()
    print(args)

    train = True
    if train:
        launch(main, args.num_gpus, args=(args,))
    else:
        main(args=args)
