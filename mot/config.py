from detectron2.config import CfgNode as CN
#from .track_registry

def add_mot_config(cfg):
    _C = cfg
    _C.MODEL.RCNN_MOT = CN()
    _C.MODEL.RCNN_MOT.NUM_CLASSES = 2
    _C.MODEL.RCNN_MOT.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.RCNN_MOT.FPN_STRIDES = [8, 16, 32, 64, 128]

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.OUT_FEATURES = ['dla2']
    _C.MODEL.DLA.USE_DLA_UP = True
    _C.MODEL.DLA.NUM_LAYERS = 34
    _C.MODEL.DLA.MS_OUTPUT = False
    _C.MODEL.DLA.NORM = 'BN'
    _C.MODEL.DLA.DLAUP_IN_FEATURES = ['dla3', 'dla4', 'dla5']
    _C.MODEL.DLA.DLAUP_NODE = 'conv'
    _C.MODEL.DLA.BACKBONE_OUT_CHANNELS = 128

    # ---------------------------------------------------------------------------- #
    # Group Norm options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.GROUP_NORM = CN()
    # Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
    _C.MODEL.GROUP_NORM.DIM_PER_GP = -1
    # Number of groups in GroupNorm (-1 if using DIM_PER_GP)
    _C.MODEL.GROUP_NORM.NUM_GROUPS = 32
    # GroupNorm's small constant in the denominator
    _C.MODEL.GROUP_NORM.EPSILON = 1e-5

    # Track head
    _C.MODEL.TRACK_ON = True
    _C.MODEL.TRACK_HEAD = CN()
    _C.MODEL.TRACK_HEAD.TRACKTOR = False
    _C.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
    _C.MODEL.TRACK_HEAD.POOLER_RESOLUTION = 15
    _C.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.TRACK_HEAD.PAD_PIXELS = 512
    # the times of width/height of search region comparing to original bounding boxes
    _C.MODEL.TRACK_HEAD.SEARCH_REGION = 2.0
    # the minimal width / height of the search region
    _C.MODEL.TRACK_HEAD.MINIMUM_SREACH_REGION = 0
    _C.MODEL.TRACK_HEAD.MODEL = 'EMM'

    # solver params
    _C.MODEL.TRACK_HEAD.TRACK_THRESH = 0.4
    _C.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.6
    _C.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4
    # maximum number of frames that a track can be dormant
    _C.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = 1

    # track proposal sampling
    _C.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE = 256
    _C.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD = 0.65
    _C.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD = 0.35

    _C.MODEL.TRACK_HEAD.IMM = CN()
    # the feature dimension of search region (after fc layer)
    # in comparison to that of target region (after fc layer)
    _C.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM_MULTIPLIER = 2
    _C.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM = 256

    _C.MODEL.TRACK_HEAD.EMM = CN()
    # Use_centerness flag only activates during inference
    _C.MODEL.TRACK_HEAD.EMM.USE_CENTERNESS = True
    _C.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.25
    _C.MODEL.TRACK_HEAD.EMM.HN_RATIO = 0.25
    _C.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT = 1.
    # The ratio of center region to be positive positions
    _C.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION = 0.8
    # The lower this weight, it allows large motion offset during inference
    # Setting this param to be small (e.g. 0.1) for datasets that have fast motion,
    # such as caltech roadside pedestrian
    _C.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT = 0.4
    #Input
    _C.INPUT.MOTION_LIMIT = 0.1
    _C.INPUT.COMPRESSION_LIMIT = 50
    _C.INPUT.MOTION_BLUR_PROB = 0.5
    _C.INPUT.AMODAL = False