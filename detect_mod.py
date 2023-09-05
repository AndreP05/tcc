# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import os
import sys
import serial
import argparse
import csv
import pandas as pd
from pathlib import Path
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.general import (check_img_size, check_imshow, check_requirements, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords)
from utils.dataloaders import LoadStreams
from models.common import DetectMultiBackend

import torch
import torch.backends.cudnn as cudnn
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler


if not os.path.exists('results/'):
    os.mkdir('results/')
    os.mkdir('results/data/')
    os.mkdir('results/frames/')
with open('results/data/data.csv', 'w', encoding='utf-8', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'scaled_xmin',
                    'scaled_ymin', 'scaled_xmax', 'scaled_ymax'])

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights_dist=ROOT / 'model@1535470106.h5',
        model_dist=ROOT / 'model@1535470106.json',
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        nosave=False,  # do not save images/videos
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    (save_dir / save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                df_test = pd.DataFrame(columns=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'scaled_xmin', 'scaled_ymin',
                                                'scaled_xmax', 'scaled_ymax'])
                object_counter = 0

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        originalvideo_size = (375, 1242)
                        originalvideo_hieght = originalvideo_size[0]
                        originalvideo_width = originalvideo_size[1]
                        img_height = im0.shape[0]
                        img_width = im0.shape[1]
                        scaled_x1 = (x1 / img_width) * originalvideo_width
                        scaled_x2 = (x2 / img_width) * originalvideo_width
                        scaled_y1 = (y1 / img_height) * originalvideo_hieght
                        scaled_y2 = (y2 / img_height) * originalvideo_hieght
                        csv_row_list = [frame, x1, y1, x2, y2, scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                        df_test.loc[object_counter] = csv_row_list
                        x_test = df_test[['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values
                        scalar = StandardScaler()
                        x_test = scalar.fit_transform(x_test)
                        scalar.fit_transform((df_test[['scaled_ymax']].values - df_test[['scaled_ymin']])/3)

                        # load json and create model
                        json_file = open(model_dist, 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()
                        loaded_model = model_from_json(loaded_model_json)

                        # load weights into new model
                        loaded_model.load_weights(weights_dist)
                        
                        # evaluate loaded model on test data
                        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
                        distance_pred = loaded_model.predict(x_test)

                        # scale up predictions to original values
                        distance_pred = scalar.inverse_transform(distance_pred)
                        annotator.box_label(
                            xyxy, f'{names[int(cls)]} { distance_pred[0]}', color=colors(int(cls), True))

                        object_counter += 1
                        with open('results/data/data.csv', 'a') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(csv_row_list)

                        cv2.imwrite('results/frames/{0}.png'.format(frame), im0)
                    
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        if 'distance_pred' in locals():
            # Process video data
            print(s, distance_pred)

            with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                ser.write("right".encode())


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--model_dist', help='model json file path')
    parser.add_argument('--weights_dist', help='model weights file path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
