#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      kien.tadinh
# @Time:        3/13/2024 9:13 PM
import cv2

COLORS = {
    'bicycle': (0, 0, 255),
    'bus': (0, 255, 255),
    'car': (255, 0, 0),
    'motorcycle': (255, 255, 0),
    'person': (0, 255, 0),
    'truck': (255, 0, 255),
}


def predict_image(model, img, img_size, conf_thres, device, inference_path, save_inference=False):
    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (img_size, img_size))
    results = model(resized_img, conf=conf_thres, verbose=False, device=device)
    if save_inference:
        txt_file = open(inference_path, "w", encoding="utf-8")
    for result in results:
        if result.boxes is None:
            continue
        for data in result.boxes.data:
            data = data.cpu().detach().numpy()
            boxes = data[:4]
            scores = data[4]
            classes = int(data[5])

            left, top, right, bottom = boxes
            left *= original_width / img_size
            top *= original_height / img_size
            right *= original_width / img_size
            bottom *= original_height / img_size

            cv2.rectangle(resized_img, (int(left), int(top)), (int(right), int(bottom)), COLORS[result.names[int(classes)]], 2)
            cv2.putText(resized_img, result.names[classes] + ": " + "{:.2f}%".format(scores * 100), (int(left), int(top)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[result.names[int(classes)]], 2)
            if save_inference:
                yolo_center_x = int((left + right) / 2)
                yolo_center_y = int((top + bottom) / 2)
                yolo_width = int(right - left)
                yolo_height = int(bottom - top)
                text_line = f"{classes} {yolo_center_x} {yolo_center_y} {yolo_width} {yolo_height} {'{:.2f}'.format(scores)}\n"
                txt_file.write(text_line)

    return img


def run_inference_in_thread(model, frame, img_size, conf, device, inference_path, save_inference, camera_idx, display_inference=False):
    img_predicted = predict_image(model, frame, img_size, conf, device, inference_path, save_inference)
    if display_inference:
        cv2.imshow(f'Camera {camera_idx}', img_predicted)
        cv2.waitKey(1)
