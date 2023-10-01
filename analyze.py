import pandas as pd 
import yaml
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np

# function input :  
# detections : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class"
# ground_truth : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class"
# function output :
# TP, FP, FN, FP
def compute_metrics(detections, ground_truth, iou_threshold=0.5) :
    # make a copy of the dataframes
    detections = detections.copy()
    ground_truth = ground_truth.copy()

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(detections)) :
        # get the detection 
        det = detections.iloc[i]
        # get the ground truth
        gt = ground_truth[ground_truth["image_name"] == det["image_name"]]
        # compute the iou between the detection and the ground truth
        iou = compute_iou(det, gt)
        # if the iou is above the threshold, it's a true positive
        if iou > iou_threshold :
            TP += 1
            # remove the ground truth from the dataframe
            ground_truth = ground_truth[ground_truth["image_name"] != det["image_name"]]
            
        # if the iou is below the threshold, it's a false positive
        else :
            FP += 1
    # compute the false negative
    FN = len(ground_truth) - TP
    # compute the true negative
    TN = len(detections) - TP
    return TP, FP, FN, TN


def compute_iou(det, gt) :
    # get the coordinates of the detection
    x1_det, y1_det, w_det, h_det = det[["x_center", "y_center", "width", "height"]]
    x2_det = x1_det + w_det
    y2_det = y1_det + h_det
    # get the coordinates of the ground truth
    x1_gt, y1_gt, w_gt, h_gt = gt[["x_center", "y_center", "width", "height"]]
    x2_gt = x1_gt + w_gt
    y2_gt = y1_gt + h_gt
    # compute the intersection
    x1_inter = max(x1_det, x1_gt)
    y1_inter = max(y1_det, y1_gt)
    x2_inter = min(x2_det, x2_gt)
    y2_inter = min(y2_det, y2_gt)
    # compute the area of the intersection
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    # compute the area of the union
    det_area = (x2_det - x1_det + 1) * (y2_det - y1_det + 1)
    gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    union_area = det_area + gt_area - inter_area
    # compute the iou
    iou = inter_area / union_area
    return iou