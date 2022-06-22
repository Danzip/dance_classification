
import torch

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np

import pytorchvideo
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection # Another option is slowfast_r50_detection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetector:
    def __init__(self,SCORE_THRESH_TEST=0.55, HUMAN_THRESH=0.75):
        self.HUMAN_THRESH = HUMAN_THRESH
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)


    def get_person_bboxes(self, inp_img):
        predictions = self.predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)

        predicted_boxes = boxes[np.logical_and(classes==0, scores>self.HUMAN_THRESH )].tensor.cpu() # only person
        return predicted_boxes
    
    def plot_bboxes(self, inp_img):
        bbox_list = self.get_person_bboxes(inp_img)
        inp_img = inp_img.numpy().astype(int)
        plt.imshow(inp_img)
        for x_1, y_1, x_2, y_2 in bbox_list:
            height = y_2 - y_1
            width = x_2 - x_1
            patch = patches.Rectangle((x_1, y_1), width,height, linewidth=1, edgecolor='r',fill=False,label='Label')
            plt.gca().add_patch(patch) 
        plt.show()
