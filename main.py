from DS import Dataset as DS
import GTSDB_toolbox
import classify

import yaml
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np

def part_1():
    working_dir = "./datasets/full/raw/"
    # load yaml file 
    with open("./params/repartitions.yml", "r") as f :
        mapping = yaml.load(f, Loader=yaml.FullLoader)

    GTSDB_toolbox.ppm_to_png(working_dir)
    # GTSDB_toolbox.format_annotation_GTSDB_to_YOLOv3(working_dir + "gt_new.txt", save=True, merge_classes=True, merge_classes_dict=mapping["map1"], normalize=True)
    # dataset = DS("./datasets/","GTSDB44",True,"./params/GTSDB44_params.yml")
    
def show_dataset_sample() :

    label_dir = "./datasets/GTSDB44/train/labels/"
    image_dir = "./datasets/GTSDB44/train/images/"

    # plot image 4 by 4 
    fig, ax = plt.subplots(2,2)
    j,k = 0,0
    for i,img in enumerate(os.listdir(image_dir)) :
        img = GTSDB_toolbox.visualize(image_dir + img, label_dir + img.split(".")[0] + ".txt")
        ax[k,j].imshow(img)
        j +=1
        if j == 2 :
            k+=1
            j=0
        if k == 2:
            plt.show()
            key = input("Press Enter to continue...")
            fig, ax = plt.subplots(2,2)
            j = 0
            k = 0
            if key == "q" :
                break

def classify_and_add_to_gt(show=False):
    df = (GTSDB_toolbox.extract_json("GTSDB.json"))
    df_new_class = df[df["object-class"] == "pn"]
    img_path = "./datasets/full/raw/"

    classify.create_fld()
    features = []
    predictions = {}
    # Create a new dataframe for the first imagename containing all the bounding boxes on the image
    for i in range(1, len(df_new_class["imagename"].unique())) :
        img_name = df_new_class["imagename"].iloc[i] 
        df_to_show = df_new_class[df_new_class["imagename"] == img_name]
        label = df_to_show[["object-class", "x_center", "y_center", "width", "height"]]
        label.columns = ["object-class", "x_center", "y_center", "width", "height"]
        path = img_path + img_name
        feature, signs = classify.color_based_feature_extraction(path, label) 
        features += feature
        predictions[path] = [len(feature),signs]
    
    kmean_labels = classify.cluster(features) # features has to be a list of tuples
    
    k = 0
    new_gt = []
    for i, path in enumerate(predictions.keys()) :
        p = predictions[path]
        img_name = path.split("/")[-1]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.clf()
        plt.subplot(1,p[0]+1,1).imshow(img)
        df_to_show = df_new_class[df_new_class["imagename"] == img_name]
        label = df_to_show[["object-class", "x_center", "y_center", "width", "height"]]
        for j,sign in enumerate(p[1]) : 
            plt.subplot(1,p[0]+1,j+2).imshow(sign)
            plt.xlabel(kmean_labels[k])

            row = label.iloc[j]
            # extract the ROI
            x_center = int(row[1])
            y_center = int(row[2])
            width = int(row[3])
            height = int(row[4])

            leftCol = x_center - width//2
            topRow = y_center - height//2
            rightCol = x_center + width//2
            bottomRow = y_center + height//2

            new_gt.append([img_name, leftCol, topRow, rightCol, bottomRow ,44-kmean_labels[k]])
            k+=1

        plt.savefig("./classify/" + img_name.split(".")[0] + "_" + str(j) + ".png")
        plt.clf()
        plt.close()

    print(new_gt)
    GTSDB_toolbox.add_new_detections_to_gt("./datasets/full/raw/gt.txt",new_gt,save=True)



if __name__ == "__main__":
    # classify_and_add_to_gt()
    part_1()
    # show_dataset_sample()