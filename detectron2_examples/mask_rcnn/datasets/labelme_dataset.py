#!/usr/bin/env python

import os
import os.path as osp
from copy import deepcopy

from typing import List
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import json
import torch

from detectron2.config import CfgNode as CN
from detectron2.data import transforms as T
from detectron2.data import detection_utils as dutils
from detectron2.structures import BoxMode

import labelme
import pycocotools.mask


@dataclass
class LabelMeDataset(object):

    root: str = None
    is_train: bool = True

    def __post_init__(self):
        assert osp.isdir(self.root)
        self._anno_path = osp.join(self.root, 'image')

        # np.random.seed(seed=256)
        self.dataset = []

    def read_content(self, path: str) -> List[str]:
        return sorted(
            [
                osp.join(self._anno_path, ifile)
                for ifile in os.listdir(path) if ifile.split('.')[-1] == 'json'
            ]
        )

    def __call__(self):
        self.dataset = []
        for index, anno_file in enumerate(
                self.read_content(self._anno_path)
        ):
            label_file = labelme.LabelFile(anno_file)
            anno = self.load_json(anno_file)
            data = dict(
                label_file=label_file,
                file_name=osp.join(self._anno_path, label_file.imagePath),
                height=anno['imageHeight'],
                width=anno['imageWidth'],
                image_id=index
            )
            self.dataset.append(data)
        return self.dataset

    def load_json(self, fn: str):
        with open(fn, 'r') as f:
            data = json.load(f)
        return data


@dataclass
class DatasetMapper(object):

    cfg: CN = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        self._augmentation = self.build_augmentation()

    def __call__(self, dataset: dict):
        dataset = deepcopy(dataset)
        label_file = dataset.pop('label_file')
        image = cv.imread(dataset['file_name'], cv.IMREAD_COLOR)
        im_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        annotations = []
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                image.shape[:2], points, shape_type
            )

            mask = np.asfortranarray(mask.astype(np.uint8))
            _mask = pycocotools.mask.encode(mask)
            bbox = pycocotools.mask.toBbox(_mask).flatten().tolist()
            area = float(pycocotools.mask.area(_mask))

            im_mask |= mask

            annotations.append(
                {
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': label,
                    'segmentation': _mask,
                    'iscrowd': 0,
                    'area': area
                }
            )
        if not self.is_train:
            return dict(
                image=image,
                annotations=annotations
            )

        aug_input = T.AugInput(image, sem_seg=im_mask)
        transforms = aug_input.apply_augmentations(self._augmentation)
        image = torch.from_numpy(
            aug_input.image.transpose((2, 0, 1)).astype('float32')
        )
        mask = torch.from_numpy(aug_input.sem_seg.astype('float32'))

        annos = [
            dutils.transform_instance_annotations(
                annotation, transforms, image.shape[1:]
            )
            for annotation in annotations
        ]

        instances = dutils.annotations_to_instances(
            annos, image.shape[1:], mask_format=self.cfg.INPUT.MASK_FORMAT
        )

        dataset['image'] = image
        dataset['sem_seg'] = mask
        dataset['instances'] = dutils.filter_empty_instances(instances)
        # dataset['annotations'] = annotations

        import IPython
        IPython.embed()
        sys.exit()

        return dataset

    def build_augmentation(self):
        result = dutils.build_augmentation(self.cfg, is_train=self.is_train)
        if self.is_train:
            result.extend(
                [
                    # T.RandomCrop('relative_range', (0.6, 0.7)),
                    T.RandomContrast(0.9, 1.1),
                    T.RandomFlip(0.5, horizontal=False, vertical=True),
                ]
            )
        return result


if __name__ == '__main__':

    import sys
    from detectron2.config import get_cfg

    r = sys.argv[1]
    l = LabelMeDataset(root=r)
    label_files = l()

    c = get_cfg()
    c.INPUT.MASK_FORMAT = 'bitmask'
    d = DatasetMapper(c, is_train=True)

    try:
        i = int(sys.argv[2])
    except:
        i = 0
    x = d(label_files[i])
    print(x)

    import matplotlib.pyplot as plt

    r = x['image'].detach().cpu().numpy() / 255.0
    m = x['sem_seg'].detach().cpu().numpy()
    r *= m

    plt.axis('off')
    plt.imshow(r.transpose([1, 2, 0]))
    plt.show()
