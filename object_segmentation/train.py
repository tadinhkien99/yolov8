#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py
# @Author:      kien.tadinh
# @Time:        11/30/2023 4:38 PM
import wandb
import yaml
from ultralytics import YOLO

wandb.init(mode="disabled")

if __name__ == "__main__":
    with open("config_train.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    project_name = config['project']
    pretrained_model = config['model']
    data = config['data']
    epochs = config['epochs']
    imgsz = config['imgsz']
    batch = config['batch']
    workers = config['workers']
    device = config['device']
    val = config['val']
    plots = config['plots']
    resume = config['resume']

    augmentation = config['augmentation']
    hsv_h = augmentation['hsv_h']
    hsv_s = augmentation['hsv_s']
    hsv_v = augmentation['hsv_v']
    degrees = augmentation['degrees']
    translate = augmentation['translate']
    scale = augmentation['scale']
    shear = augmentation['shear']
    perspective = augmentation['perspective']
    flipud = augmentation['flipud']
    fliplr = augmentation['fliplr']
    mosaic = augmentation['mosaic']
    mixup = augmentation['mixup']
    copy_paste = augmentation['copy_paste']
    erasing = augmentation['erasing']
    crop_fraction = augmentation['crop_fraction']

    model = YOLO(pretrained_model)
    results = model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, device=device,
                          project=project_name, val=val, plots=plots, resume=resume, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, degrees=degrees, translate=translate,
                          scale=scale, shear=shear, perspective=perspective, flipud=flipud, fliplr=fliplr, mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
                          erasing=erasing, crop_fraction=crop_fraction)
