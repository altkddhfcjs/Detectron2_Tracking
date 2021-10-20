from detectron2.config import get_cfg

# Detectron2 Train library
import datetime
import time
import logging
import torch

# Training
from mot.config import add_mot_config
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer

# Detectron
from detectron2 import model_zoo
from detectron2.structures import Instances, Boxes
from detectron2.modeling import build_model

#siam
from siam.configs.defaults import cfg as siam_cfg
from siam.data.build_train_data_loader import build_train_data_loader
from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler

# do_train funtion
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
import os

WINDOW_NAME = "CenterNet2 detections"

logger = logging.getLogger("detectron2")


def setup_cfg(path=None):
    cfg = get_cfg()
    add_mot_config(cfg)
    #add_centernet_config(cfg)
    #cfg.merge_from_file(path)

    if path == None:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.META_ARCHITECTURE = "RCNN_MOT"

    cfg.DATASETS.TRAIN = ("crowdhuman_train_fbox",)# "coco_2017_train")
    cfg.DATASETS.TEST = ("crowdhuman_val_fbox",)
    cfg.DATASETS.ROOT_DIR = "datasets/"
    cfg.MODEL.DEVICE = 'cuda:1'
    cfg.OUTPUT_DIR = "output2/MOT_DLA"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.TRACK_ON = True
    cfg.MODEL.TRACK_TEST = False

    ########## SOLVER config setting ###############
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.WEIGHT_DECAY_BIAS = 0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.11111111
    cfg.DATALOADER.MODALITY = "image"
    ################################################

    if cfg.MODEL.TRACK_ON:
        cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.25
        cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
        cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.05 ##predictor
        cfg.INPUT.TO_BGR255 = False
        cfg.DATALOADER.SIZE_DIVISIBILITY = 32
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.VIS_PERIOD = 1

    return cfg

def setup_siam_cfg(cfg, config_file):
    """
    cfg: Detectron2 config file
    config_file: SiamTracker config file
    """
    detectron_backbone = cfg.MODEL.BACKBONE.NAME
    siam_backbone = ""
    output_channels = cfg.MODEL.FPN.OUT_CHANNELS

    siam_cfg.merge_from_file(config_file)
    siam_cfg.DATASETS.ROOT_DIR = "./datasets"
    if "dla" in detectron_backbone and "fpn" in detectron_backbone:
        siam_backbone = "DLA-{}-FPN".format(cfg.MODEL.DLA.NUM_LAYERS)
        siam_cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS = output_channels

    if "res" in detectron_backbone and "fpn" in detectron_backbone:
        siam_backbone = "R-{}-FPN".format(cfg.MODEL.RESNETS.DEPTH)
        siam_cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = output_channels

    siam_cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1
    siam_cfg.MODEL.DEVICE = cfg.MODEL.DEVICE
    siam_cfg.MODEL.BACKBONE.CONV_BODY = siam_backbone
    siam_cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH
    #siam_cfg.DATASETS.TRAIN = ("crowdhuman_train_fbox", ) #"COCO17_train", )
    siam_cfg.DATASETS.TRAIN = ("COCO17_train",)

    siam_cfg.freeze()
    return siam_cfg


def do_train(siam_cfg, cfg, model, resume=True):
    model.train()
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer.load(cfg.MODEL.WEIGHTS)

    start_iter = (
            checkpointer.resume_or_load(
                cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    ##### Using Siam dataloader #####
    data_loader = build_train_data_loader(siam_cfg, is_distributed=False, shuffle=False)
    siam_dataloader = True
    #################################

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            #################################################3
            ## Detectron2 data format
            ## list
            ##   -list
            ##      -dict
            ##          "file_name"
            ##          "image_id"
            ##          "height"
            ##          "width"
            ##          "image"
            ##          "instance"
            ##              - gt_boxes
            ##              - gt_classes
            ##              - gt_ids
            #######################################################
            if siam_dataloader:
                images = data[0]
                gts = data[1]
                file_names = data[2]
                image_ids = data[3]
                data = []
                for i in range(len(image_ids)):
                    video = []
                    video_name = file_names[i]
                    video_gt = gts[(i*2):(i*2)+2]
                    video_frame = images[(i*2):(i*2)+2]
                    video_id = image_ids[i]
                    c, v_h, v_w = video_frame[0].shape
                    for v in range(2):
                        gt = video_gt[v]
                        bbox = Boxes(gt.bbox)
                        labels = gt.get_field("labels") - 1
                        ids = gt.get_field("ids")
                        w, h = gt.size

                        instance = Instances((h, w))
                        instance.set("gt_boxes", bbox)
                        instance.set("gt_classes", labels)
                        instance.set("gt_ids", ids)
                        frame = {
                            "file_name": video_name,
                            "image_id": video_id,
                            "height": v_h,
                            "width": v_w,
                            "image": video_frame[v],
                            "instances": instance
                        }
                        video.append(frame)
                    data.append(video)

            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

cfg = setup_cfg()
cfg.MODEL.WEIGHTS = "output2/MOT_RCNN_SIAM_SAMPLER/model_0024999.pth"
siam_cfg = setup_siam_cfg(cfg, "config/dla/DLA_34_FPN_EMM.yaml")

model = build_model(cfg)
do_train(siam_cfg, cfg, model, resume=True)
