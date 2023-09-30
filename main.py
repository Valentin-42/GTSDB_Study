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
    GTSDB_toolbox.format_annotation_GTSDB_to_YOLOv3(working_dir + "gt.txt", save=True, merge_classes=False, merge_classes_dict=mapping["3_classes"], normalize=True)
    dataset = DS("./datasets/","GTSDB",True,"./params/GTSDB_params.yml")
    
def show_dataset_sample() :

    label_dir = "./datasets/GTSDB/train/labels/"
    image_dir = "./datasets/GTSDB/train/images/"

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

def part_2(show=False):
    df = (GTSDB_toolbox.extract_json("GTSDB.json"))
    df_new_class = df[df["object-class"] == "pn"]
    img_path = "./datasets/full/raw/"

    classify.create_fld()

    # Create a new dataframe for the first imagename containing all the bounding boxes on the image
    for i in range(1, len(df_new_class["imagename"].unique())) :

        img_name = df_new_class["imagename"].iloc[i] 

        df_to_show = df_new_class[df_new_class["imagename"] == img_name]
        label = df_to_show[["object-class", "x_center", "y_center", "width", "height"]]
        label.columns = ["object-class", "x_center", "y_center", "width", "height"]
        if(show) :
            img = GTSDB_toolbox.visualize(img_path, label)
            # plot the image
            plt.imshow(img)
            plt.show()
            input("Press Enter to continue...")
            plt.close()
        path = img_path + img_name
        classify.extract_ROIS(path, label,"./classify/")
    
    classify.get_average_size_color_shape("./classify/")
    
    features = []
    for img in os.listdir("./classify/") :
        features.append(classify.extract_hog_features("./classify/" + img).tolist())
    print(features)
    classify.cluster(features)



if __name__ == "__main__":
    # part_1()
    # show_dataset_sample()
    part_2()