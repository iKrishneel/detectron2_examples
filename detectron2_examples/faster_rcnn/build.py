#!/usr/bin/env python

from dataclasses import dataclass
from typing import Collection
import argparse

from torch.optim import Optimizer

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data.build import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler
from detectron2.utils.events import EventStorage

from detectron2_examples.faster_rcnn import Dataloader, DatasetMapper
from detectron2_examples.faster_rcnn.car196 import Car196


def register_dataset(
        dataset_root: str, dataset_name: str, data_type: str = 'train'
):
    DatasetCatalog.register(
        dataset_name, Car196(root=dataset_root, data_type=data_type)
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=[
            "car",
        ],
        thing_color=[
            (0, 255, 0),
        ],
    )


def register_datasets(
        dataset_roots: Collection[str], dataset_names: Collection[str],
        data_type: str = 'train'
):
    for dataset_root, dataset_name in zip(dataset_roots, dataset_names):
        register_dataset(
            dataset_root, dataset_name, data_type=data_type
        )


def read_all_dataset_dicts(dataset_names: Collection[str]):
    assert len(dataset_names)

    dataset_name_to_dicts = {}
    for dataset_name in dataset_names:
        dataset_name_to_dicts[dataset_name] = DatasetCatalog.get(dataset_name)


@dataclass(init=True)
class Trainer(DefaultTrainer):

    cfg: CfgNode = None

    def __post_init__(self):
        super(Trainer, self).__init__(self.cfg)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):

        mapper = DatasetMapper(cfg, True) if mapper is None else mapper
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        pass

    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer: Optimizer):
        return build_d2_lr_scheduler(cfg, optimizer)


def add_faster_rcnn_config(cfg, args):
    cfg.INPUT.ROTATION_ANGLES = 10
    cfg.OUTPUT_DIR = args.output_dir

    try:
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError:
        pass


def setup(args, ):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_faster_rcnn_config(cfg, args)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    register_datasets(
        [
            args.root,
        ],
        cfg.DATASETS.TRAIN,
    )
    read_all_dataset_dicts(cfg.DATASETS.TRAIN)

    trainer = Trainer(cfg)
    # trainer.build_model(cfg)
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
