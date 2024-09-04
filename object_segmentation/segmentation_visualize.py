#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    segmentation_visualize.py
# @Author:      Kuro
# @Time:        9/4/2024 12:16 AM

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_image(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    segments = []
    cls_list = []
    with open(label_path, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            cls = int(coords[0])
            points = np.array(coords[1:], dtype=np.float32).reshape(-1, 2)
            segments.append(points * [w, h])
            cls_list.append(cls)

    for i in range(len(segments)):
        color = (0, 255, 0) if cls_list[i] == 0 else (0, 0, 255)
        cv2.polylines(img, [segments[i].astype(np.int32)], isClosed=True, color=color, thickness=7)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # image_path = "data/train/images/1_jpg.rf.7734eb7fedd7525f112b2586eddc13f1.jpg"
    # label_path = "data/train/labels/1_jpg.rf.7734eb7fedd7525f112b2586eddc13f1.txt"
    image_path = "augmented/images/1_jpg.rf.7734eb7fedd7525f112b2586eddc13f1.jpg"
    label_path = "augmented/labels/1_jpg.rf.7734eb7fedd7525f112b2586eddc13f1.txt"
    visualize_image(image_path, label_path)
