#!/usr/bin/env python

from dataclasses import dataclass
from copy import deepcopy

import torch

from detectron2.config import CfgNode as CN
from detectron2.data import transforms as T
from detectron2.data import detection_utils as dutils


@dataclass
class DatasetMapper(object):

    cfg: CN = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        self._augmentation = self.build_augmentation()

    def __call__(self, dataset_dict: dict):
        dataset_dict = deepcopy(dataset_dict)

        image = dutils.read_image(
            dataset_dict['file_name'], format=self.cfg.INPUT.FORMAT
        )
        # dutils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = aug_input.apply_augmentations(self._augmentation)
        image = torch.from_numpy(
            aug_input.image.transpose((2, 0, 1)).astype("float32")
        )

        if not self.is_train:
            return dict(image=image)

        annotations = dataset_dict.pop('annotations', None)
        assert annotations

        annos = [
            dutils.transform_instance_annotations(
                annotation, transforms, image.shape[1:]
            )
            for annotation in annotations
        ]
        instances = dutils.annotations_to_instances(annos, image.shape[1:])

        dataset_dict['image'] = image
        dataset_dict['instances'] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def build_augmentation(self):

        result = dutils.build_augmentation(self.cfg, is_train=self.is_train)
        if self.is_train:
            random_rotation = T.RandomRotation(
                self.cfg.INPUT.ROTATION_ANGLES,
                expand=False,
                sample_style='choice',
            )

            # resize = T.Resize((800, 800))
            result.extend(
                [
                    random_rotation,
                ]
            )
        return result


if __name__ == '__main__':

    import sys
    from detectron2.config import get_cfg

    from car196 import Car196

    car = Car196(root=sys.argv[1])

    cfg = get_cfg()
    cfg.INPUT.ROTATION_ANGLES = 90

    d = DatasetMapper(cfg=cfg, is_train=True)

    s = car()[1000]
    print(s)
    x = d(s)
    print(x)
