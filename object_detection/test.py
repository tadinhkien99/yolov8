#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Author:      kien.tadinh
# @Time:        11/30/2023 5:06 PM
import time

import yaml
from ultralytics import YOLO

if __name__ == "__main__":
    with open("config_test.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    project = config['project']
    model_path = config['model']
    source = config['source']
    imgsz = config['imgsz']
    device = config['device']
    conf = config['conf']
    iou = config['iou']
    save = config['save']
    save_txt = config['save_txt']
    show = config['show']
    show_labels = config['show_labels']
    show_conf = config['show_conf']
    show_boxes = config['show_boxes']

    model = YOLO(model_path)
    start_time = time.time()
    model.predict(verbose=False, project=project, source=source, imgsz=imgsz, device=device, conf=conf, iou=iou, save=save, save_txt=save_txt,
                  show=show, show_labels=show_labels, show_conf=show_conf, boxes=show_boxes)
    print("Predict time {} seconds".format(time.time() - start_time))
