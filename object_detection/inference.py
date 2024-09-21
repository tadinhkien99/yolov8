#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    inference_main.py
# @Author:      kien.tadinh
# @Time:        3/20/2024 10:27 AM
import argparse
import os
import time
from datetime import datetime

import cv2
import yaml
from ultralytics import YOLO

from model import predict_image
from utils import create_directory


def main(inference_folder, model_checkpoint, prefix=''):
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    server = config['server']
    server_ip = server['ip']
    server_port = server['port']
    server_username = server['username']
    server_password = server['password']
    server_recheck = server['recheck']
    model_config = config['model']
    img_size = model_config['size']
    conf = model_config['conf']
    device = model_config['device']
    station = config['station_Name']
    dock = config['dock']
    send_pic_to_server = config['send_pic_to_server']
    inference = config['inference']
    inference_show = config['inference_show']
    save_pic = config['save_picture']
    save_inference = config['save_inference']

    create_directory(inference_folder)

    model = YOLO(model_checkpoint)
    buffer_size = 4

    while True:
        # print("Procesing....")
        # print(time.time())
        current_dateTime = datetime.now()
        # print(current_dateTime)
        buffer_count = 0
        frames = {}
        all_files = os.listdir(inference_folder)
        total_images = len([file for file in all_files if not file.endswith('.txt')])
        for idx, file in enumerate(os.listdir(inference_folder)):
            if file.endswith('txt'):
                continue
            file_name_without_ext = os.path.splitext(file)[0]
            # Check whether folder don't have file + 'prefix' + .txt file
            inference_file_name = f'{prefix}{file_name_without_ext}.txt'
            if not os.path.exists(f'{inference_folder}/{inference_file_name}'):
                buffer_count += 1
                frames[file_name_without_ext] = cv2.imread(f'{inference_folder}/{file}')
            if buffer_count == buffer_size or idx == total_images - 1:

                for file_name, frame in frames.items():
                    inference_path = f'{inference_folder}/{prefix}{file_name}.txt'
                    # Create a Thread for each inference call
                    try:
                        img_predicted = predict_image(model, frame, img_size, conf, device, inference_path, save_inference)
                    except:
                        print("error infernce")

                frames = {}
        print(buffer_count)
        time.sleep(0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inference_folder', type=str, help='Folder to save inference results')
    parser.add_argument('--checkpoint', type=str, help='Model path')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix for saving inference results')
    args = parser.parse_args()

    main(args.inference_folder, args.checkpoint, args.prefix)
