from detectron2.config import get_cfg
import time
from mot.config import add_mot_config
from detectron2.engine import DefaultPredictor
from detectron2.data import transforms as T
from detectron2 import model_zoo
from demos.utils.vis_generator import VisGenerator

import cv2
import os

import argparse

parser = argparse.ArgumentParser(" SiamMOT Inference Demo")
parser.add_argument('--demo-video', metavar="FILE", type=str,
                    required=True)
parser.add_argument("--output-path", type=str, default=None,
                    help='The path of dumped videos')
parser.add_argument("--model", type=str, default='models/res_101_fpn_mot.pth',
                    help='The path of model weight file')
parser.add_argument("--det-confidence", type=float, default=0.75,
                    help='Detector model output confidence')
parser.add_argument("--track-confidence", type=float, default=0.75,
                    help='Confidence value of create new track')

def setup_cfg():
    cfg = get_cfg()
    add_mot_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.META_ARCHITECTURE = "RCNN_MOT"

    cfg.DATASETS.TRAIN = ("crowdhuman_train_fbox",)
    cfg.DATASETS.TEST = ("crowdhuman_val_fbox",)
    cfg.DATASETS.ROOT_DIR = "datasets/"
    cfg.MODEL.DEVICE = 'cuda:0'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.TRACK_ON = True
    cfg.MODEL.TRACK_TEST = False

    #cfg.MODEL.
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.DATALOADER.MODALITY = "image"

    ##tracking setting
    if cfg.MODEL.TRACK_ON:
        cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.15
        cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
        cfg.INPUT.TO_BGR255 = False
        cfg.DATALOADER.SIZE_DIVISIBILITY = 32
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.VIS_PERIOD = 1

    return cfg

def demo_inference(predictor, cap, writer):
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        orig_frame = frame[:, :, ::-1]
        start_time = time.time()
        output = predictor(orig_frame)
        inference_time = time.time() - start_time
        vis_frame = vis_generator.frame_vis_generator(frame, output, frame_id=frame_id, speed=inference_time)
    
        writer.write(vis_frame)
        cv2.imshow("result", vis_frame)
        key = cv2.waitKey(100)
        if key == ord('c'):
           break
        frame_id += 1
    
if __name__ == '__main__':
    args = parser.parse_args()
    cfg = setup_cfg()

    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_confidence
    cfg.MODEL.TRACK_HEAD.TRACK_THRESH = 0.3
    cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = args.track_confidence
    cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4
    vis_generator = VisGenerator(vis_height=1080)

    cfg.MODEL.TRACK_ON = True
    cfg.MODEL.TRACK_HEAD.TRACKTOR = True

    predictor = DefaultPredictor(cfg)
    predictor.model.reset_mot_status()

    video_path = args.demo_video
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "output.avi")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    demo_inference(predictor, cap, writer)
    
    writer.release()
