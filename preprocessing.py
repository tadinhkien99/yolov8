#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    preprocessing.py
# @Author:      kien.tadinh
# @Time:        3/3/2024 4:10 PM

import os
import shutil

from tqdm import tqdm

if __name__ == "__main__":

    data_path = "train"
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")

    data_folder_name = "dataset"

    if not os.path.exists(data_folder_name):
        os.mkdir(data_folder_name)

    # Split data 70-20-10
    for folder in ["train", "valid"]:
        if not os.path.exists(os.path.join(data_folder_name, "images")):
            os.mkdir(os.path.join(data_folder_name, "images"))
        if not os.path.exists(os.path.join(data_folder_name, "labels")):
            os.mkdir(os.path.join(data_folder_name, "labels"))
        if not os.path.exists(os.path.join(data_folder_name, "images", folder)):
            os.mkdir(os.path.join(data_folder_name, "images", folder))
        if not os.path.exists(os.path.join(data_folder_name, "labels", folder)):
            os.mkdir(os.path.join(data_folder_name, "labels", folder))

    # Move images and labels to dataset
    img_list = os.listdir(images_path)
    training_size = int(len(img_list) * 0.8)
    # valid_size = int(len(img_list) * 0.2)
    # test_size = len(img_list) - training_size - valid_size
    valid_size = len(img_list) - training_size

    for i, img in tqdm(enumerate(img_list), total=len(img_list)):
        if i < training_size:
            shutil.copy(os.path.join(images_path, img), os.path.join(data_folder_name, "images", "train", img))
            shutil.copy(os.path.join(labels_path, img.replace(".jpg", ".txt")), os.path.join(data_folder_name, "labels", "train", img.replace(".jpg", ".txt")))
        else:
            shutil.copy(os.path.join(images_path, img), os.path.join(data_folder_name, "images", "valid", img))
            shutil.copy(os.path.join(labels_path, img.replace(".jpg", ".txt")), os.path.join(data_folder_name, "labels", "valid", img.replace(".jpg", ".txt")))
        # else:
        #     shutil.copy(os.path.join(images_path, img), os.path.join(data_folder_name, "images", "test", img))
        #     shutil.copy(os.path.join(labels_path, img.replace(".jpg", ".txt")), os.path.join(data_folder_name, "labels", "test", img.replace(".jpg", ".txt")))
