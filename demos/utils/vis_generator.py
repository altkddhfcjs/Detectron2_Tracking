import random

import cv2
import numpy as np
from matplotlib import cm

from detectron2.data import transforms as T


class VisGenerator:
    """
    Generate a video for visualization
    """
    def __init__(self, vis_height=None):
        """
        vis_height is the resolution of output frame
        """
        self._vis_height = vis_height
        # by default, 50 colors
        self.num_colors = 50
        self.colors = self.get_n_colors(self.num_colors)
        # use coco class name order
        #self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                       'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                       'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                       'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                       'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    @staticmethod
    def get_n_colors(n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors

    def normalize_output(self, frame, results):
        frame_height, frame_width = frame.shape[:2]
        rescale_ratio = float(self._vis_height) / float(frame_height)
        new_height = int(round(frame_height * rescale_ratio))
        new_width = int(round(frame_width * rescale_ratio))

        aug = T.ScaleTransform(frame_height, frame_width, new_height, new_width, interp="bilinear")
        frame = cv2.resize(frame, (new_width, new_height))
        #frame = aug.apply_image(frame)

        if self._vis_height is not None and results is not None:
            boxlist_height = results.image_size[0]
            assert (boxlist_height == frame_height)
            bbox = results.pred_boxes.tensor
            bbox = aug.apply_box(bbox)
            results.set("pred_boxes", bbox)

        return frame, results

    def frame_vis_generator(self, frame, results, frame_id=None, speed=0):
        if len(results) == 0:
            frame, _ = self.normalize_output(frame, None)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            results = results['instances'].to("cpu")
            frame, results = self.normalize_output(frame, results)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ids = results.get('ids')
            idx = ids >= 0
            if np.count_nonzero(idx) > 0:
                if len(results) == 1:
                    results = results
                else:
                    results = results[idx]

                bbox = results.pred_boxes #.detach().cpu().numpy() #results.pred_boxes[ids >= 0]
                ids = results.get('ids').tolist()
                labels = results.get('pred_classes').tolist()

                for i, entity_id in enumerate(ids):
                    color = self.colors[entity_id % self.num_colors]
                    class_name = self.class_names[labels[i]]
                    text_width = len(class_name) * 20
                    x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(np.int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

                    cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

                    # Draw black background rectangle for test
                    cv2.rectangle(frame, (x1 - 5, y1 - 25), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

        if frame_id != None:
            cv2.putText(frame, str(frame_id), (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, [10, 50, 255], thickness=3)

        if speed > 0:
            fps = float(1 / speed)
            cv2.putText(frame, "FPS: {:.3}".format(fps), (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [10, 50, 255], thickness=3)
        return frame
