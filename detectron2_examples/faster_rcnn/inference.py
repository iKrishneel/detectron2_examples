#!/usr/bin/env python

import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from detectron2.engine.defaults import DefaultPredictor
from detectron2_examples.faster_rcnn import setup


def main(args=None):

    from car196 import Car196

    car196 = Car196(args.root, data_type='test')
    car196()
    # loader = Dataloader(car196, is_test=True)

    cfg = setup(args=args)
    predictor = DefaultPredictor(cfg)

    for index in range(len(car196)):
        index = 200
        image, target = car196.get(index)
        r = predictor(image)

        try:
            instances = r['instances'].to('cpu').get_fields()
            bboxes = instances['pred_boxes'].tensor.numpy()
            scores = instances['scores'].numpy()
            labels = instances['pred_classes'].numpy()

            bboxes = bboxes.astype(np.intp)

            for bbox, score, label in zip(bboxes, scores, labels):
                if score < 0.5 or label != 0:
                    continue

                x1, y1, x2, y2 = bbox
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.imwrite('result.png', image)

        except Exception as e:
            print("Nothing found: ", index, e)

        break

    # import IPython
    # IPython.embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", metavar="FILE")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--root", required=True, type=str)
    args = parser.parse_args()

    main(args=args)
