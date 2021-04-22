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
        image = torch.from_numpy(aug_input.image.transpose((2, 0, 1)).astype("float32"))
        
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
        instances = dutils.annotations_to_instances(
            annos, image.shape[1:]
        )

        """
        import cv2 as cv
        import matplotlib.pyplot as plt
        import numpy as np
        x1,y1,x2,y2 = np.array(annos[0]['bbox']).astype(np.intp)
        print(x1, y1, x2, y2, image.shape)
        image = image.permute((1, 2, 0)).numpy() / 255.0
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.rectangle(image, (x1, y1, x2-x1, y2-y1), (0, 255, 0), 3)
        plt.imshow(image)
        plt.show()
        """
        return {
            'image': image,
            'instances': instances
        }
        
    def build_augmentation(self):
        
        result = dutils.build_augmentation(self.cfg, is_train=self.is_train)
        if self.is_train:
            random_rotation = T.RandomRotation(
                self.cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style='choice'
            )
            
            resize = T.Resize((800, 800))
            result.extend([random_rotation, resize])
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
    d(s)
    
