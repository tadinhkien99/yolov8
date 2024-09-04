#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    augmentation.py
# @Author:      Kuro
# @Time:        9/3/2024 11:34 PM
import argparse
import os
import random

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/train', help='Path to the data directory')
parser.add_argument('--save_dir', default='augmented', help='Path to save directory')
args = parser.parse_args()


def masks2segments(masks, strategy='largest'):
    segments = []
    for x in masks:
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    return cv2.resize(mask, (nw, nh))


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [np.array(segments[si]).reshape(-1, 2)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
    return ms


def read_image(img_path, label_path):
    img = cv2.imread(img_path)
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
    return img, segments, cls_list


def augmentation_transform(img, masks, aug_method):
    """Augmentation transform."""
    # A.HorizontalFlip(p=1)
    masks = np.stack(masks, axis=0)
    transform = A.Compose(aug_method, is_check_shapes=False)
    transformed = transform(image=img, masks=masks)
    return transformed['image'], transformed['masks']


if __name__ == "__main__":
    data_dir = args.data_dir
    save_dir = args.save_dir
    img_data_dir = os.path.join(data_dir, 'images')
    label_data_dir = os.path.join(data_dir, 'labels')
    img_save_dir = os.path.join(save_dir, 'images')
    label_save_dir = os.path.join(save_dir, 'labels')
    all_aug_methods = [
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.GridDistortion(p=1),
        A.ElasticTransform(p=1),
        A.CoarseDropout(p=1),
        A.Blur(p=1),
        A.Rotate(p=1),
        A.RandomBrightnessContrast(p=1),
    ]

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    for img_filename in tqdm(os.listdir(img_data_dir)):
        img_path = os.path.join(img_data_dir, img_filename)
        label_path = os.path.join(label_data_dir, img_filename.rsplit('.', 1)[0] + '.txt')

        img, segments, cls_list = read_image(img_path, label_path)
        original_h, original_w = img.shape[:2]
        masks = polygons2masks_overlap((original_h, original_w), segments)

        n_augmethods = random.randint(1, 3)
        aug_methods = random.sample(all_aug_methods, n_augmethods)
        img_aug, masks_aug = augmentation_transform(img, masks, aug_methods)
        img_save_path = os.path.join(img_save_dir, img_filename.rsplit('.', 1)[0] + '_augment.jpg')
        label_save_path = os.path.join(label_save_dir, img_filename.rsplit('.', 1)[0] + '_augment.txt')
        cv2.imwrite(img_save_path, img_aug)

        # Convert masks to segments
        segments_aug = masks2segments(masks_aug)
        with open(label_save_path, 'w') as f:
            for i in range(len(segments_aug)):
                cls = cls_list[i]
                segment_normalize = segments_aug[i] / [original_w, original_h]
                f.write(f"{cls} {' '.join(map(str, segment_normalize.flatten()))}\n")
