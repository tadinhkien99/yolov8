import argparse
import os
import platform
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

COLORS = {
    'bicycle': (0, 0, 255),
    'bus': (0, 255, 255),
    'car': (255, 0, 0),
    'motorcycle': (255, 255, 0),
    'person': (0, 255, 0),
    'truck': (255, 0, 255),
}


def detect(opt):
    yolo_weights, source, out, imgsz, conf_thres, view_vid, save_vid, device, save_txt = (
        opt.weights, opt.source, opt.output, opt.img_size, opt.conf_thres,
        opt.view_vid, opt.save_vid, opt.device, opt.save_txt)

    if not os.path.exists(out):
        os.makedirs(out)

    model = YOLO(yolo_weights)
    vid_cap = cv2.VideoCapture(source)

    t0 = time.time()

    save_path = str(Path(out))
    filename = Path(source).name

    raw_txt_path = str(Path(out)) + "/RAW_" + filename[:-4] + ".txt"

    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_path = os.path.join(save_path, filename[:-4] + ".avi")

    if save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    if save_txt:
        txt_file = open(raw_txt_path, "w", encoding="utf-8")
    #         txt_initial = f"Video File:{source}\nwidth:{w}\nheight:{h}\nfps:{fps}\n"
    #         txt_args = f"Model weight:{model_wt_name}\nImage size:{imgsz}\nstart\n"

    inferenced_count = 1
    pbar = tqdm(total=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        _, frame = vid_cap.read()
        image_data = cv2.resize(frame, (imgsz, imgsz))
        results = model(image_data, conf=conf_thres, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            for data in result.boxes.data:
                data = data.cpu().detach().numpy()
                boxes = data[:4]
                scores = data[4]
                classes = data[5]

                left, top, right, bottom = boxes
                yolo_center_x = (left + right) / 2
                yolo_center_y = (top + bottom) / 2
                yolo_width = right - left
                yolo_height = bottom - top

                cv2.rectangle(image_data, (int(left), int(top)), (int(right), int(bottom)), COLORS[result.names[int(classes)]], 2)
                cv2.putText(image_data, result.names[int(classes)] + ": " + "{:.2f}%".format(scores * 100), (int(left), int(top)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[result.names[int(classes)]], 2)

                text_line = f"{int(classes)},{yolo_center_x},{yolo_center_y},{yolo_width},{yolo_height},{inferenced_count},{'{:.2f}'.format(scores)}\n"
                txt_file.write(text_line)

        # Stream results
        if view_vid:
            cv2.imshow("Inference", image_data)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        inferenced_count += 1

        if save_vid:
            # image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            image_data = cv2.resize(image_data, (w, h))
            vid_writer.write(image_data)
        pbar.update(1)

    if save_txt or save_vid:
        print("Results saved to %s" % os.getcwd() + os.sep + out)
        txt_file.close()
        if platform == "darwin":  # MacOS
            os.system("open " + save_path)
    print("Done. (%.3fs)" % (time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="vehicle_train/train6/weights/best.pt", help="model.pt path")
    parser.add_argument("--source", default="C:/temp/tt.mp4", type=str, required=False, help="source")
    parser.add_argument("--output", type=str, default="C:/temp/", help="output folder")  # output folder
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.6, help="object confidence threshold")
    parser.add_argument("--view-vid", default=False, action="store_true", help="View video results")
    parser.add_argument("--save-vid", default=False, action="store_true", help="Save video results")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-txt", default=True, action="store_true", help="save results to *.txt")

    args = parser.parse_args()
    detect(args)
