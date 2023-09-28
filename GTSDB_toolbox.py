import os 
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Load annotation csv file
# 00000.ppm;774;411;815;446;11 is an example of a line in the csv file
# imagename;#leftCol#;#topRow#;#rightCol#;#bottomRow#;#ClassID is the format
# ClassID name is in the file classes.txt
# turn the csv file into a dataframe and convert it to another dataframe with the following format : 
# <image name> <object-class> <x_center> <y_center> <width> <height>
# add a save argument to save labels in a txt file with the following format : <object-class> <x_center> <y_center> <width> <height>
# the name of the txt file should be the same as the image name
# create a folder called label in the same directory as the images folder and save the txt files in it
# if it already exists, delete it and create a new one
# verify that the number of images and the number of txt files are the same
# merge_classes_dict should be a dict like {'class-id-to-replace' : 'replacement'}
def format_annotation_GTSDB_to_YOLOv3(annotation_csv_path, save=False, img_ext = "ppm", merge_classes = False, merge_classes_dict = None, normalize = False) :
    df = pd.read_csv(annotation_csv_path, sep=";", header=None)
    df.columns = ["imagename", "leftCol", "topRow", "rightCol", "bottomRow", "ClassID"]
    df["object-class"] = df["ClassID"]
    df["width"] = df["rightCol"] - df["leftCol"]
    df["height"] = df["bottomRow"] - df["topRow"]
    if normalize :
        w,h = cv2.imread(os.path.join(os.path.dirname(annotation_csv_path), df["imagename"].iloc[0]).replace("\\", "/")).shape[:2]
        df["leftCol"] = df["leftCol"] / h
        df["rightCol"] = df["rightCol"] / h
        df["topRow"] = df["topRow"] / w
        df["bottomRow"] = df["bottomRow"] / w 
        df["width"] = df["width"] / w
        df["height"] = df["height"] / h
        df["image_width"] = w
        df["image_height"] = h
        
    df["x_center"] = (df["leftCol"] + df["rightCol"]) / 2
    df["y_center"] = (df["topRow"] + df["bottomRow"]) / 2

    df = df[["imagename", "object-class", "x_center", "y_center", "width", "height"]]

    if save :
        # create a folder called label in the same directory as the images folder and save the txt files in it
        # if it already exists, delete it and create a new one
        if os.path.exists(os.path.join(os.path.dirname(annotation_csv_path), "labels")) :
            shutil.rmtree(os.path.join(os.path.dirname(annotation_csv_path), "labels"))
        os.mkdir(os.path.join(os.path.dirname(annotation_csv_path), "labels"))
        # save labels in a txt file with the following format : <object-class> <x_center> <y_center> <width> <height>
        for img_name in df["imagename"].unique() :
            img_df = df[df["imagename"] == img_name]
            img_df = img_df[["object-class", "x_center", "y_center", "width", "height"]]
            if merge_classes and merge_classes_dict != None :
                img_df["object-class"] = img_df["object-class"].replace(merge_classes_dict)
            img_df.to_csv(os.path.join(os.path.dirname(annotation_csv_path), "labels", img_name.split(".")[0] + ".txt").replace("\\", "/"), sep=" ", header=False, index=False)
        
        # verify that the number of images and the number of txt files are the same
        if len(os.listdir(os.path.join(os.path.dirname(annotation_csv_path), "labels"))) != len(df["imagename"].unique()) :
            print("Error : the number of images and the number of txt files are not the same : {} images and {} txt files".format(len(os.listdir(os.path.dirname(annotation_csv_path))), len(os.listdir(os.path.join(os.path.dirname(annotation_csv_path), "labels")))))
            return None
        
        # verify that the number of raw images and the number of txt files are the same
        print(f"{len(os.listdir(os.path.dirname(annotation_csv_path))) - len(df['imagename'].unique())} empty images")
        if len(os.listdir(os.path.dirname(annotation_csv_path))) != len(df["imagename"].unique()) :
            for img_name in os.listdir(os.path.dirname(annotation_csv_path)) :
                if img_name.split(".")[-1] == img_ext :
                    if img_name.split(".")[0] + ".txt" not in os.listdir(os.path.join(os.path.dirname(annotation_csv_path), "labels")) :
                        print(f"{img_name} has no annotations")
                        with open(os.path.join(os.path.dirname(annotation_csv_path), "labels", img_name.split(".")[0] + ".txt").replace("\\", "/"), "w") as f :
                            pass
        
    print("Labels created  : {} images and {} txt files".format(len([f for f in os.listdir(os.path.dirname(annotation_csv_path)) if f.endswith(img_ext)]), len(os.listdir(os.path.join(os.path.dirname(annotation_csv_path), "labels")))))
    return df
 

# create a function that plots the bounding boxes on an image
# given a dataframe with the following format : <image name> <object-class> <x_center> <y_center> <width> <height>
# images paths can be found using the root argument of the function and the images names in the dataframe
# it should display 4 images per 4 images by creating a subplots and wait for the user to press a key to display the for other images 
def plot_bounding_boxes(df, root="./") :
    # create a list of images paths
    images_paths = []
    for img_name in df["imagename"].unique() :
        images_paths.append(os.path.join(root, img_name))
    
    # create a list of images
    images = []
    for img_path in images_paths :
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)

    
    # create a list of images with bounding boxes
    images_with_bounding_boxes = []
    for img, img_name in zip(images, df["imagename"].unique()) :
        # get the bounding boxes for the image
        img_df = df[df["imagename"] == img_name]
        for _, row in img_df.iterrows() :
            x_center = int(row["x_center"])
            y_center = int(row["y_center"])
            width = int(row["width"])
            height = int(row["height"])
            cv2.rectangle(img, (x_center - width//2, y_center - height//2), (x_center + width//2, y_center + height//2), (0, 255, 0), 2)
        images_with_bounding_boxes.append(img)
    
    # display the images with bounding boxes
    # Wait for the user to press a key to display the for other images
    for i in range(0, len(images_with_bounding_boxes), 4) :
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(images_with_bounding_boxes[i])
        axs[0, 1].imshow(images_with_bounding_boxes[i+1])
        axs[1, 0].imshow(images_with_bounding_boxes[i+2])
        axs[1, 1].imshow(images_with_bounding_boxes[i+3])
        plt.show()
        plt.close()
        cmd = input("[Enter] continue, [s] save, [q] quit")
        if cmd == "q" :
            break
        if cmd == "s" :
            plt.savefig("img_set_{}.png".format(i))

    return images_with_bounding_boxes

# create a function that convert ppm to png
# given a folder with ppm images
# it should create a folder with png images
def ppm_to_png(ppm_folder_path) :
    # create a folder with png images
    png_folder_path = os.path.join(ppm_folder_path, "png")
    if os.path.exists(png_folder_path) :
        shutil.rmtree(png_folder_path)
    os.mkdir(png_folder_path)

    # convert ppm to png
    for img_name in os.listdir(ppm_folder_path) :
        if img_name.split(".")[-1] == "ppm" :
            img = cv2.imread(os.path.join(ppm_folder_path, img_name))
            cv2.imwrite(os.path.join(png_folder_path, img_name.split(".")[0] + ".png"), img)
    
    print("Images converted from ppm to png")
    return None