#!/usr/bin/env python

import time
from collections import Counter
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data import detection_utils, DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler
from detectron2.engine.defaults import AMPTrainer

from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances


class AccumGradAMPTrainer(AMPTrainer):

    def __init__(self, accumulate: int = 1, *args, **kwargs):
        super(AccumGradAMPTrainer, self).__init__(*args, **kwargs)

        assert accumulate > 0, f'Accumulate must be >= 1 {accumulate}'
        self._accumulate = accumulate

    def run_step(self):
        assert self.model.training, 'model was changed to eval mode!'
        assert torch.cuda.is_available(), 'CUDA is required for AMP Training'

        start = time.perf_counter()
        data_time = time.perf_counter() - start

        loss_dicts = {}
        self.optimizer.zero_grad()

        for index in range(self._accumulate):
            data = next(self._data_loader_iter)
            loss_dict = self._run_one_step(data=data)
            if index == 0:
                loss_dicts = Counter(loss_dict)
                continue
            loss_dicts += Counter(loss_dict)

        self._write_metrics(loss_dicts, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def _run_one_step(self, data) -> dict:
        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {
                    'total_loss': loss_dict
                }
            else:
                losses = sum(loss_dict.values())

        self.grad_scaler.scale(losses).backward()
        return loss_dict


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        aug = detection_utils.build_augmentation(cfg, is_train=cls.is_train)
        if cls.is_train:
            aug.extend(
                [
                    T.RandomContrast(0.9, 1.1),
                    T.RandomFlip(0.5, horizontal=False, vertical=True),
                ]
            )
        mapper = DatasetMapper(cfg, cls.is_train, augmentations=aug)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_schedule(cls, cfg: CfgNode, optim: Optimizer):
        return build_d2_lr_scheduler(cfg, optim)


def add_mask_rcnn_config(cfg, args):
    cfg.OUTPUT_DIR = args.output_dir

    try:
        assert args.weights is not None
        cfg.MODEL.WEIGHTS = args.weights
    except (AttributeError, AssertionError) as e:
        print('Not weight provided: ', e)


def setup(args) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_mask_rcnn_config(cfg, args)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    parser.add_argument('--json', required=False, default='trainval.json')
    parser.add_argument('--name', required=False, default=None, type=str)
    args = parser.parse_args()

    if args.name is not None:
        register_coco_instances(args.name, {}, args.json, args.root)

    train = True
    if train:
        launch(main, args.num_gpus, args=(args,))
    else:
        main(args=args)
