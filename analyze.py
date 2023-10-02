import pandas as pd 
import yaml
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil

# function input :  
# detections : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class"
# # ground_truth : pandas df with columns : "image_name", "x_center", "y_center", "width", "height", "class"
# function output :
# TP,TN,FP,FN as ints 
# A dt dataframe which update the detections df with the type of each detection appended the last column
def compute_metrics(detections, ground_truth, im_name) :

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    detections = detections.copy()
    ground_truth = ground_truth.copy()
    ground_truth.reset_index(drop=True)

    # if there are no detections and no ground truths, all are true negatives
    if len(detections) == 0 and len(ground_truth) == 0 :
        print("No detections and no ground truths")
        TN = len(detections)
        row = [im_name, -1, -1, -1, -1, "none", 1]
        detections = pd.concat([detections, pd.DataFrame([row], columns=["image_name", "x_center", "y_center", "width", "height", "class", "type"])], ignore_index=True)
        return TP,TN, FP, FN , detections, ground_truth

    # if there are no detections, all the ground truths are false negatives
    if len(detections) == 0 :
        print("No detections")
        FN = len(ground_truth)
        for index_gt, gt in ground_truth.iterrows() :
            row = [im_name, -1, -1, -1, -1, "none", 3]
            detections = pd.concat([detections, pd.DataFrame([row], columns=["image_name", "x_center", "y_center", "width", "height", "class", "type"])], ignore_index=True)
        return TP, TN, FP, FN, detections, ground_truth
    
    # if there are no ground truths, all the detections are false positives
    if len(ground_truth) == 0 :
        print("No ground truths")
        FP = len(detections)
        for index_det, dt in detections.iterrows() :
            detections.at[index_det, "type"] = 2
        return TP,TN, FP, FN, detections, ground_truth
    
    # ELSE : there are detections and ground truths
    # for each detection
    for index_det, det in detections.iterrows() :
        # get the image name
        image_name = det["image_name"]
        # for each ground truth
        print(ground_truth)
        for index_gt, gt in ground_truth.iterrows() :
            # compute the iou between the detection and the ground truth
            iou = compute_iou(det, gt)
            # if iou > 0.5, it's a true positive
            if iou > 0.5 :
                TP += 1
                print(f"TP : {iou}%")
                # update the detections dataframe
                detections.at[index_det, "type"] = 0
                # remove the ground truth from the dataframe
                ground_truth = ground_truth.drop(index_gt)
                # stop the loop
                break
    

    # the remaining detections are false positives
    # loop in the detections dataframe to find the one that have no type and set their type to 2
    # check if type column exists for the det 
    for index_det, det in detections.iterrows() :
        # check if type column exists for the det 
        if "type" not in det :
            FP +=1
            detections.at[index_det, "type"] = 2
    
    for index_gt, gt in ground_truth.iterrows() :
        FN += 1
        row = [gt["image_name"], -1, -1, -1, -1, "none", 3]
        detections = pd.concat([detections, pd.DataFrame([row], columns=["image_name", "x_center", "y_center", "width", "height", "class", "type"])], ignore_index=True)


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
def plot_and_save(detections, ground_truth,img_fld_path ,output_path) :
    
        show = False
        print("Saving at : ", output_path)
        # create tp tn fp fn folders in output path
        tp_path = os.path.join(output_path, "tp").replace("\\","/")
        tn_path = os.path.join(output_path, "tn").replace("\\","/")
        fp_path = os.path.join(output_path, "fp").replace("\\","/")
        fn_path = os.path.join(output_path, "fn").replace("\\","/")
        if os.path.exists(tp_path) :
            shutil.rmtree(tp_path)
        os.makedirs(tp_path)
        if os.path.exists(tn_path) :
            shutil.rmtree(tn_path)
        os.makedirs(tn_path)
        if os.path.exists(fp_path) :
            shutil.rmtree(fp_path)    
        os.makedirs(fp_path)
        if os.path.exists(fn_path) :
            shutil.rmtree(fn_path)
        os.makedirs(fn_path)

        # get the list of images
        images = detections["image_name"].unique()
        # for each image
        for image_name in images :
            TP,TN,FP,FN = 0,0,0,0

            image_path = os.path.join(img_fld_path, image_name).replace("\\","/")
            print(image_path)
            image_type = []
            # get the detections for this image
            dets = detections[detections["image_name"] == image_name]
            # get the ground truths for this image
            gts = ground_truth[ground_truth["image_name"] == image_name]
            # read the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_dt = image.copy()
            image_gt = image.copy()
            image_h, image_w, _ = image.shape
            # plot the detections
            for index_det, det in dets.iterrows() :
                # get the coordinates of the detection
                x1, y1, w, h = det[["x_center", "y_center", "width", "height"]]
                x2 = x1 + w//2
                y2 = y1 + h//2
                x1 = x1 - w//2
                y1 = y1 - h//2 
                # get the class of the detection
                class_name = det["class"]
                # get the type of the detection
                type_det = det["type"]
                # choose the color of the detection
                if type_det == 0 :
                    color = (0, 255, 0)
                    txt = "TP"
                    TP +=1
                elif type_det == 2 :
                    color = (0, 0, 255)
                    txt = "FP"
                    FP +=1
                elif type_det == 3 :
                    txt = "FN"
                    FN +=1
                elif type_det == 1 :
                    txt = "TN"
                    TN +=1

                if txt == "TP" or txt == "FP":
                    # draw the detection
                    cv2.rectangle(image_dt, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # write the class next to the detection
                    cv2.putText(image_dt, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # write the type next to the detection
                    cv2.putText(image_dt, txt, (int(x1) + 50, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                image_type.append(type_det)
            # plot the ground truths
            for index_gt, gt in gts.iterrows() :
                # get the coordinates of the ground truth
                x1, y1, w, h = gt[["x_center", "y_center", "width", "height"]]
                x2 = x1 + w//2
                y2 = y1 + h//2
                x1 = x1 - w//2
                y1 = y1 - h//2
                # draw the ground truth
                cv2.rectangle(image_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # show the image
            fig, axs = plt.subplots(1, 2, figsize=(30, 20))

            axs[0].imshow(image_gt)
            axs[0].axis('off')
            axs[0].set_title("Groundtruths", fontsize=40)

            axs[1].imshow(image_dt)
            axs[1].axis('off')
            axs[1].set_title(f"Detections : TP : {TP}, TN : {TN}, FP : {FP}, FN : {FN}", fontsize=40)

            if show :
                plt.show()
                k = input("Press [ENTER] : Continue | [q] : Cancel | [p] : Stop plotting >> ")
                if k == "q" :
                    exit()
                if k == "p" :
                    show = False

            # save the image
            plt.tight_layout()
            if 0 in image_type :
                plt.savefig(os.path.join(tp_path, os.path.basename(image_name)))
            elif 1 in image_type :
                plt.savefig(os.path.join(tn_path, os.path.basename(image_name)))
            elif 2 in image_type :
                plt.savefig(os.path.join(fp_path, os.path.basename(image_name)))
            elif 3 in image_type :
                plt.savefig(os.path.join(fn_path, os.path.basename(image_name)))
            plt.close()



def analyze() :
    working_dir = "./analyze/40/"

    predictions_fld = os.path.join(working_dir, "predict","labels").replace("\\","/")
    ground_truth_fld = os.path.join(working_dir, "predict","gt").replace("\\","/")
    image_fld = os.path.join(working_dir, "predict", "images").replace("\\","/")
    # for all images in the predictions folder
    # load all ground truths in a pandas dataframe 
    # columns : "image_name", "x_center", "y_center", "width", "height", "class" 
   
    gts = []
    dts = []
    for im in os.listdir(image_fld) :
        if im.endswith(".png") :
            image_path = os.path.join(image_fld, im).replace("\\","/")
            gt_path = image_path.replace("images","gt").replace(".png",".txt")
            dt_path = image_path.replace("images","labels").replace(".png",".txt")

            # if gt path file has content
            if os.path.getsize(gt_path) > 0 :
                with open(gt_path, "r") as f :
                    for line in f.readlines() :
                        line = line.split(" ")
                        class_name = line[0]
                        x_center = float(line[1])
                        y_center = float(line[2])
                        width = float(line[3])
                        height = float(line[4])

                        # denormalize if needed
                        if x_center < 1 and y_center < 1 and width < 1 and height < 1 :
                            temp = cv2.imread(image_path)
                            h, w, _ = temp.shape
                            x_center *= w
                            y_center *= h
                            width *= w
                            height *= h
                        gts.append([im,x_center,y_center,width,height,class_name])
            
            if os.path.exists(dt_path) > 0 :
                with open(dt_path, "r") as f :
                    for line in f.readlines() :
                        line = line.split(" ")
                        class_name = line[0]
                        x_center = float(line[1])
                        y_center = float(line[2])
                        width = float(line[3])
                        height = float(line[4])

                        # denormalize if needed
                        if x_center < 1 and y_center < 1 and width < 1 and height < 1 :
                            temp = cv2.imread(image_path)
                            h, w, _ = temp.shape
                            x_center *= w
                            y_center *= h
                            width *= w
                            height *= h
                        dts.append([im,x_center, y_center,width,height,class_name])

    ground_truth = pd.DataFrame(gts, columns=["image_name", "x_center", "y_center", "width", "height", "class"])
    detections = pd.DataFrame(dts, columns=["image_name", "x_center", "y_center", "width", "height", "class"])
    # compute the metrics
    print(ground_truth)
    print(detections)
    typed_detection = pd.DataFrame(columns=["image_name", "x_center", "y_center", "width", "height", "class", "type"])
    for im in os.listdir(image_fld) :
        dt = detections[detections["image_name"] == im]
        gt = ground_truth[ground_truth["image_name"] == im]
        TP, TN, FP, FN, d,g = compute_metrics(dt, gt, im)
        print(f"TP : {TP}, TN : {TN}, FP : {FP}, FN : {FN}")
        typed_detection = pd.concat([typed_detection, d], ignore_index=True)
    print(typed_detection)

    # plot the detections and the ground truths
    raw_images = os.path.join("./datasets/","full","raw","png").replace("\\","/")
    plot_and_save(typed_detection, ground_truth,raw_images ,os.path.join(working_dir, "output").replace("\\","/"))

if __name__ == "__main__" :
    analyze()