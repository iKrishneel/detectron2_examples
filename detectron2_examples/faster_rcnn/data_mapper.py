#!/usr/bin/env python

from copy import deepcopy

import torch

from detectron2.data import transforms as T
from detectron2.data import detection_utils as dutils


class DatasetMapper(object):

    def __init__(self, cfg, is_train: bool = True):
        self._cfg = cfg
        self._is_train = is_train
        
        self._augmentation = self.build_augmentation()

    def __call__(self, dataset_dict: dict):
        print(dataset_dict)
        dataset_dict = deepcopy(dataset_dict)

        image = dutils.read_image(
            dataset_dict['file_name'], format=self._cfg.INPUT.FORMAT
        )
        # dutils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self._augmentation, image)
        image_shape = image.shape[:2]
        image = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )

        if not self._is_train:
            return dict(image=image)

        # TODO: annotation
        
    def build_augmentation(self):
        cfg = self._cfg
        result = dutils.build_augmentation(cfg, is_train=self._is_train)
        if self._is_train:
            random_rotation = T.RandomRotation(
                cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style='choice'
            )
            # resize = T.Resize((800, 800))
            result.extend([random_rotation])
        return result
    

if __name__ == '__main__':

    from detectron2.config import get_cfg
    cfg = get_cfg()

    cfg.INPUT.ROTATION_ANGLES = 10
    d = DatasetMapper(cfg, True)
