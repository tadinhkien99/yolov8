#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      kien.tadinh
# @Time:        3/13/2024 9:13 PM
import random

import cv2
import numpy as np
from ultralytics import YOLO


def overlay(image, mask, color, alpha, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def predict_image(model, img, img_size, conf_thres, device, inference_path, save_inference=False, colors=None):
    original_height, original_width = img.shape[:2]
    resize_img = cv2.resize(img, (img_size, img_size))
    results = model(resize_img, conf=conf_thres, verbose=False, device=device, imgsz=img_size)
    if save_inference:
        txt_file = open(inference_path, "w", encoding="utf-8")
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segment masks outputs

        if masks is not None:
            for mask, box in zip(masks, boxes):
                seg = mask.data[0].cpu().numpy()
                seg = cv2.resize(seg, (original_width, original_height))
                img = overlay(img, seg, colors[int(box.cls)], 0.4)
                xmin = int(box.data[0][0] * original_width / img_size)
                ymin = int(box.data[0][1] * original_height / img_size)
                xmax = int(box.data[0][2] * original_width / img_size)
                ymax = int(box.data[0][3] * original_height / img_size)
                cls = int(box.cls)
                plot_one_box([xmin, ymin, xmax, ymax], img, colors[cls], f'{class_names[cls]} {float(box.conf):.3}')
                if save_inference:
                    if mask.xy[0].shape[0] == 0:
                        continue
                    txt_line = str(int(box.cls))
                    for point in mask.xy[0]:
                        x = point[0] * original_width / img_size
                        y = point[1] * original_height / img_size
                        # convert to 0-1 scale
                        x = x / original_width
                        y = y / original_height
                        txt_line += f' {x} {y}'
                    txt_file.write(txt_line + '\n')

        # img = cv2.resize(img, (816, 816))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    return img


def run_inference_in_thread(model, frame, img_size, conf, device, inference_path, save_inference, camera_idx, display_inference=False):
    img_predicted = predict_image(model, frame, img_size, conf, device, inference_path, save_inference)
    if display_inference:
        cv2.imshow(f'Camera {camera_idx}', img_predicted)
        cv2.waitKey(1)


if __name__ == "__main__":
    img = cv2.imread(r'E:\kuro\test\onviz\yolov8-series\object_detection\IMG_20240211_112309.jpg')
    model = YOLO(r'E:\kuro\test\onviz\yolov8-series\object_detection\weels.pt')
    class_names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
    predict_image(model, img, 640, 0.5, 'cpu', 'inference.txt', save_inference=True, colors=colors)
