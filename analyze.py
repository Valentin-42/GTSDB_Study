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
# TP,TN,FP,FN as ints 
# A dt dataframe which update the detections df with the type of each detection appended the last column
def compute_metrics(detections, ground_truth) :

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    detections = detections.copy()
    ground_truth = ground_truth.copy()

    # if there are no detections and no ground truths, all are true negatives
    if len(detections) == 0 and len(ground_truth) == 0 :
        TN = len(detections)
        for index_det, det in detections.iterrows() :
            detections.at[index_det, "type"] = "TN"
        return TP, FP, FN, TN

    # if there are no detections, all the ground truths are false negatives
    if len(detections) == 0 :
        FN = len(ground_truth)
        for index_gt, gt in ground_truth.iterrows() :
            ground_truth.at[index_gt, "type"] = "FN"
        return TP, FP, FN
    
    # if there are no ground truths, all the detections are false positives
    if len(ground_truth) == 0 :
        FP = len(detections)
        return TP, FP, FN
    
    # ELSE : there are detections and ground truths
    # for each detection
    for index_det, det in detections.iterrows() :
        # get the image name
        image_name = det["image_name"]
        # get the ground truths for this image
        gts = ground_truth[ground_truth["image_name"] == image_name]
        # for each ground truth
        for index_gt, gt in gts.iterrows() :
            # compute the iou between the detection and the ground truth
            iou = compute_iou(det, gt)
            # if iou > 0.5, it's a true positive
            if iou > 0.5 :
                TP += 1
                print(f"TP : {iou}%")
                # update the detections dataframe
                detections.at[index_det, "type"] = "TP"
                # remove the ground truth from the dataframe
                ground_truth = ground_truth.drop(gt.index)
                # stop the loop
                break
    
    # the remaining ground truths are false negatives
    FN = len(ground_truth)
    for index_gt, gt in ground_truth.iterrows() :
        ground_truth.at[index_gt, "type"] = "FN"

    return TP, TN, FP, FN, detections, ground_truth

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

# method input : 
# detections : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class", "type"
# ground_truth : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class"
# create a subplot for each image showing the detections and the ground truths 
# TP detection should be in green, FP in red, FN in blue, TN in orange
# class of the detection should be written next to the detection
# ground truths should be plotted next to the detection in a subplot
def plot(detections, ground_truth) :
    
        # get the list of images
        images = detections["image_name"].unique()
    
        # for each image
        for image_name in images :
            # get the detections for this image
            dets = detections[detections["image_name"] == image_name]
            # get the ground truths for this image
            gts = ground_truth[ground_truth["image_name"] == image_name]
            # read the image
            image = cv2.imread(image_name)
            image_dt = image.copy()
            image_gt = image.copy()
            # plot the detections
            for index_det, det in dets.iterrows() :
                # get the coordinates of the detection
                x1, y1, w, h = det[["x_center", "y_center", "width", "height"]]
                x2 = x1 + w
                y2 = y1 + h
                # get the class of the detection
                class_name = det["class"]
                # get the type of the detection
                type_det = det["type"]
                # choose the color of the detection
                if type_det == "TP" :
                    color = (0, 255, 0)
                elif type_det == "FP" :
                    color = (0, 0, 255)
                elif type_det == "FN" :
                    color = (255, 0, 0)
                elif type_det == "TN" :
                    color = (0, 165, 255)
                # draw the detection
                cv2.rectangle(image_dt, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # write the class next to the detection
                cv2.putText(image_dt, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # plot the ground truths
            for index_gt, gt in gts.iterrows() :
                # get the coordinates of the ground truth
                x1, y1, w, h = gt[["x_center", "y_center", "width", "height"]]
                x2 = x1 + w
                y2 = y1 + h
                # draw the ground truth
                cv2.rectangle(image_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # show the image
            plt.subplot(1,2,1).imshow(image_gt)
            plt.subplot(1,2,2).imshow(image_dt)
            plt.axes('off')
            plt.title(image_name)
            plt.show()
            k = input("Press any key to continue")
            if k == "q" :
                exit()