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

    model = YOLO(pretrained_model)
    results = model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, device=device,
                          project=project_name, val=val, plots=plots, resume=resume)
