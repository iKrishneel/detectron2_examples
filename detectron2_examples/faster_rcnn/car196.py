#!/usr/bin/env python

from dataclasses import dataclass

import os
import numpy as np

from typing import List, Dict, Tuple

import scipy.io as sio
# from PIL import Image
import cv2 as cv


@dataclass
class Car196(object):

    root: str = None
    data_type: str = 'train'

    def __post_init__(self):
        assert os.path.isdir(self.root)

        data_images = os.path.join(self.root, f'cars_{self.data_type}')
        assert os.path.isdir(data_images)

        files = sorted(os.listdir(data_images))
        self.annotations = sio.loadmat(
            os.path.join(self.root, f'devkit/cars_{self.data_type}_annos.mat')
        )['annotations'][0]

        self.image_paths = [
            os.path.join(data_images, f)
            for f in files
            if f.split('.')[-1] in ['jpg', 'png']
        ]

        assert len(self.image_paths) == len(self.annotations)
        self.dataset = None

    def __call__(self, index: int = None) -> List[Dict]:

        self.dataset = []
        for index in range(len(self.image_paths)):
            try:
                x1, y1, x2, y2, labels, fn = self.annotations[index]
            except ValueError:
                x1, y1, x2, y2, fn = self.annotations[index]
                labels = [[0]]
                
            bboxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
            labels = labels[0]
            annotations = [
                {
                    'bbox': bbox,
                    'bbox_mode': 0,
                    'category_id': 0,  # assume only car is needed
                }
                for bbox, label in zip(bboxes, labels)
            ]
            data = dict(
                file_name=self.image_paths[index],
                image_id=index + 1,
                annotations=annotations,
                height=800,
                width=800,
            )
            self.dataset.append(data)
        return self.dataset

    def get(self, index):
        # assert self.dataset and index < self.dataset
        dataset_dict = self.dataset[index]

        # image = Image.open(dataset_dict['file_name'])
        image = cv.imread(dataset_dict['file_name'], cv.IMREAD_ANYCOLOR)
        return image, dataset_dict
    
    def __len__(self):
        return len(self.image_paths)


def show(image: np.ndarray):

    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':

    import sys

    c = Car196(root=sys.argv[1])
    print(c())
