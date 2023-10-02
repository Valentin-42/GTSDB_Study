# --root
#      |--DatasetName
#           |--train  -- ratio_train
#                |--images
#                |--labels
#           |--val    -- ratio_val
#                |--images
#                |--labels
#           |--test   -- ratio_test
#                |--images
#                |--labels

import os
import pandas as pd
import yaml 
import shutil
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class Dataset() :
    def __init__(self, root=None, name=None,create_from_scratch=False, yaml_path=None) :
        
        if root == None :
            root = "./"

        if name == None :
            name = "Dataset0"
        # Paths
        self.root = root
        self.name = name

        self.train_fld_name = "train"
        self.val_fld_name   = "val"
        self.test_fld_name  = "test"
        self.info_fld_name = "info"

        self.images_fld_name = "images"
        self.labels_fld_name = "labels"

        self.labels_extension_ftype = ".txt"
        self.imgs_extension_ftype   = ".png"
                
        self.paths = []
        self.image_paths = []
        self.labels_paths = []
        
        self.__set_dataset_train_path()
        self.__set_dataset_val_path()
        self.__set_dataset_test_path()
        self.__set_dataset_info_path()

        if create_from_scratch == True and yaml_path != None :
            self.unbalanced_class_tolerance = 5 # min Number of img per class
            print(f"Creating dataset from scratch using the yaml file : {yaml_path}")
            self.get_params(yaml_path)
            self.create_dataset(root)

        self.check_paths()

        print(f"Dataset {self.name} created with {self.number_of_images} images : {self.number_of_train_images} in train set, {self.number_of_val_images} in val set and {self.number_of_test_images} in test set")
    
    def __set_dataset_train_path(self) :
        parent_folder = os.path.join(self.root, self.name)
        self.train_path = os.path.join(parent_folder, "train").replace("\\", "/")
        self.train_images_path = os.path.join(self.train_path, self.images_fld_name).replace("\\", "/")
        self.train_labels_path = os.path.join(self.train_path, self.labels_fld_name).replace("\\", "/")
        
    def __set_dataset_val_path(self) :
        self.val_path        = os.path.join(self.root, self.name, "val").replace("\\", "/")
        self.val_images_path = os.path.join(self.val_path, self.images_fld_name).replace("\\", "/")
        self.val_labels_path = os.path.join(self.val_path, self.labels_fld_name).replace("\\", "/")

    def __set_dataset_test_path(self) :
        self.test_path        = os.path.join(self.root, self.name, "test").replace("\\", "/")
        self.test_images_path = os.path.join(self.test_path, self.images_fld_name).replace("\\", "/")
        self.test_labels_path = os.path.join(self.test_path, self.labels_fld_name).replace("\\", "/")

    def __set_dataset_info_path(self) :
        self.info_path = os.path.join(self.root, self.name, self.info_fld_name).replace("\\", "/")

    def check_paths(self) :
        self.paths.extend([self.root, self.train_path, self.train_images_path, self.train_labels_path, self.val_path, self.val_images_path, self.val_labels_path, self.test_path, self.test_images_path, self.test_labels_path])
        self.image_paths.extend([self.train_images_path, self.val_images_path, self.test_images_path])
        self.labels_paths.extend([self.train_labels_path, self.val_labels_path, self.test_labels_path])  

        for path in self.paths :
            if not os.path.exists(path) :
                raise Exception("Path {} does not exist".format(path))

        for path in self.image_paths :
            for im in os.listdir(path) :
                if not im.endswith(self.imgs_extension_ftype) :
                    raise Exception("Image {} does not have the right extension".format(im))
        
        for path in self.labels_paths :
            for lb in os.listdir(path) :
                if not lb.endswith(self.labels_extension_ftype) :
                    raise Exception("Label {} does not have the right extension".format(lb))       
    
    def set_images_extension_ftype(self, ext) :
        self.imgs_extension_ftype = ext

    def set_labels_extension_ftype(self, ext) :
        self.labels_extension_ftype = ext
    
    # Create a dataset from scratch
    # root : path to the dataset folder
    # - the name of the dataset
    # - the ratio of images in the train set
    # - the ratio of images in the val set
    # - the ratio of images in the test set
    # - the classes names file path
    # - the folder path containing the imagess
    def create_dataset(self, root) :
       
        # filling the train val and test set
        self.__create_sets()

    # Create the train, val and test sets
    # raw_folder : path to the folder containing images and labels
    def __create_sets(self) :
        # get the images and labels paths
        images_paths = []
        labels_paths = []
        for im in os.listdir(self.raw_images_path) :
            if im.endswith(self.imgs_extension_ftype) :
                images_paths.append(os.path.join(self.raw_images_path, im))
        for lb in os.listdir(self.raw_labels_path) :
            if lb.endswith(self.labels_extension_ftype) :
                labels_paths.append(os.path.join(self.raw_labels_path, lb))
        
        # check if the images and labels have the same name
        images_names = [os.path.basename(im).split(".")[0] for im in images_paths]
        labels_names = [os.path.basename(lb).split(".")[0] for lb in labels_paths]
        if len(images_names) != len(labels_names) :
            raise Exception("Images and labels do not have the same name: {} != {}".format(len(images_names), len(labels_names)))
                
        # check if every image has a label
        if len(images_paths) != len(labels_paths) :
            raise Exception("Some images do not have a label")

        self.df_dt, self.df_stats, self.empty_images = self.__sets_stats(images_paths, labels_paths)
        self.__create_fld_architecture()
        self.__plot_stats(False)
        self.__create_class_uniform_train_val_test()

    # Sets must contain equivalent number of classes and size of bounding boxes
    # Load the labels txt files in a dataframe
    # each label name correspond to the image name and lines correspond to <object-class> <x_center> <y_center> <width> <height> 
    # compute the mean and std of the width and height of the bounding boxes
    # compute the mean and std of the number of classes per image
    # compute the mean and std of the number of bounding boxes per image
    # compute the mean and std of the number of bounding boxes per class
    # compute the mean and std of the size and area of bounding boxes per class
    # add this dataframe to self
    # create the train, val and test sets using the dataframe and the computed mean and std
    def __sets_stats(self, images_paths, labels_paths) :
        # create a dataframe containing the labels
        lst = {}
        empty_images = []
        for lb_path in labels_paths :
            imagename = os.path.basename(lb_path).split(".")[0] + self.imgs_extension_ftype
            if os.path.getsize(lb_path) > 0 :
                f = open(lb_path, "r")
                lines = f.readlines()
                f.close()
                lines = [line.strip() for line in lines]
                lst[imagename] = [ l.split(' ') for l in lines ]
            else :
                empty_images.append(os.path.basename(lb_path).split(".")[0] + self.imgs_extension_ftype)
           
        # Create an empty list to hold rows of the final dataframe
        data = []
        # Iterate through the dictionary and convert data to the desired format
        for imagename, object_data in lst.items():
            # open image to get its height and width
            image = cv2.imread(os.path.join(self.raw_images_path, imagename))
            im_height, im_width, _ = image.shape
            for item in object_data:
                object_class, x_center, y_center, width, height = item
                if float(x_center) < 1 and float(y_center) < 1 and float(width) < 1 and float(height) < 1 :
                    x_center = int(float(x_center) * im_width)
                    y_center = int(float(y_center) * im_height)
                    width    = int(float(width) * im_width)
                    height   = int(float(height) * im_height)
                data.append([imagename, object_class, x_center, y_center, width, height])

        df_detections = pd.DataFrame(data, columns=["imagename", "object-class", "x_center", "y_center", "width", "height"])
        df_detections = df_detections.sort_values(by="imagename")

        # compute the mean and std of the width and height of the bounding boxes
        df_detections["width"] = df_detections["width"].astype(float)
        df_detections["height"] = df_detections["height"].astype(float)
        mean_width = df_detections["width"].mean()
        std_width = df_detections["width"].std()
        mean_height = df_detections["height"].mean()
        std_height = df_detections["height"].std()
        mean_area = (df_detections["width"] * df_detections["height"]).mean()
        std_area = (df_detections["width"] * df_detections["height"]).std()
        print("mean_width  : {}".format(round(mean_width,2)))
        print("std_width   : {}".format(round(std_width,2)))
        print("mean_height : {}".format(round(mean_height,2)))
        print("std_height  : {}".format(round(std_height,2)))
        print("mean_area   : {}".format(round(mean_area,2)))
        print("std_area    : {}".format(round(std_area,2)))

        # compute the mean and std of the number of classes per image
        classes_per_image = df_detections.groupby("imagename")["object-class"].nunique()
        mean_classes_per_image = classes_per_image.mean()
        std_classes_per_image = classes_per_image.std()
        print("mean_classes_per_image : {}".format(round(mean_classes_per_image,2)))
        print("std_classes_per_image  : {}".format(round(std_classes_per_image,2)))

        # compute the mean and std of the number of bounding boxes per image
        bbs_per_image = df_detections.groupby("imagename")["object-class"].count()
        mean_bbs_per_image = bbs_per_image.mean()
        std_bbs_per_image = bbs_per_image.std()
        print("mean_bbs_per_image : {}".format(round(mean_bbs_per_image,2)))
        print("std_bbs_per_image : {}".format(round(std_bbs_per_image,2)))

        # compute the mean and std of the number of bounding boxes per class
        bbs_per_class = df_detections.groupby("object-class")["object-class"].count()
        mean_bbs_per_class = bbs_per_class.mean()
        std_bbs_per_class = bbs_per_class.std()
        print("mean_bbs_per_class : {}".format(round(mean_bbs_per_class,2)))
        print("std_bbs_per_class : {}".format(round(std_bbs_per_class,2)))

        # compute the mean and std of the size and area of bounding boxes per class
        mean_width_per_class = df_detections.groupby("object-class")["width"].mean()
        std_width_per_class  = df_detections.groupby("object-class")["width"].std() 
        mean_height_per_class = df_detections.groupby("object-class")["height"].mean()
        std_height_per_class  = df_detections.groupby("object-class")["height"].std()
        mean_area_per_class   = df_detections.groupby("object-class")["width"].mean() * df_detections.groupby("object-class")["height"].mean()
        std_area_per_class    = df_detections.groupby("object-class")["width"].std() * df_detections.groupby("object-class")["height"].std()

        df_stats = pd.DataFrame()
        df_stats["nbs_bbs_per_class"]     = bbs_per_class
        df_stats["mean_width_per_class"]  = mean_width_per_class
        df_stats["std_width_per_class"]   = std_width_per_class
        df_stats["mean_height_per_class"] = mean_height_per_class
        df_stats["std_height_per_class"]  = std_height_per_class
        df_stats["mean_area_per_class"]   = mean_area_per_class
        df_stats["std_area_per_class"]    = std_area_per_class
        
        return df_detections, df_stats, empty_images

    # Use defined dataframes to plot :
    # Plot the repartition of the classes along the images
    # Plot the repartition of the size of the bounding boxes along the classes
    # Plot the repartition of the size of the bounding boxes along the images
    # Plot the repartition of the number of bounding boxes along the classes
    def __plot_stats(self, display=True) :
        
        # Plot an sample image of each class
        # Create a dictionary of class names and a corresponding image and detection
        class_images = {}
        for class_name in self.df_dt["object-class"].unique() :
            class_images[class_name] = self.df_dt[self.df_dt['object-class'] == class_name].to_dict(orient='records')[0]
        # sort by object-class
        class_images = dict(sorted(class_images.items(), key=lambda item: int(item[0])))
        # Create a grid subplot to display all classes bases on length of class_images
        n = int(np.sqrt(len(class_images))) + 1
        fig, axes = plt.subplots(n, n, figsize=(7, 7))
        fig.suptitle("Sample Images of Each Class")
        j,k=0,0
        for i, (class_name, elmt) in enumerate(class_images.items()) :
            im_path = os.path.join(self.raw_images_path, elmt['imagename'])
            image = cv2.imread(im_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x_center, y_center, width, height = int(float(elmt['x_center'])), int(float(elmt['y_center'])), int(float(elmt['width'])), int(float(elmt['height']))

            axes[k,j].imshow(image[y_center - height//2:y_center + height//2, x_center - width//2:x_center + width//2])
            axes[k,j].set_title(f'Class {class_name}', fontsize=8, y=1.0, pad=1)
            j += 1
            if j>=n :
                j=0
                k+=1

            print(j,k)

        for p in range(0, n) :
            for q in range(0, n) :
                axes[p,q].axis('off')

        if display == True :
            plt.show()

        plt.savefig(os.path.join(self.info_path, "sample_images.png").replace("\\", "/"))
        plt.close()

        # Plot the repartition of the classes along the images
        class_counts = self.df_dt.value_counts("object-class")
        plt.bar(class_counts.index, class_counts.values)
        plt.title("Repartition of the classes along the images")
        plt.xlabel("Classes")
        plt.ylabel("Number of images")
        plt.xticks(rotation=45)

        if display == True :
            plt.show()

        plt.savefig(os.path.join(self.info_path, "class_repartition.png").replace("\\", "/"))
        plt.close()

        # For class under the tolerance, plot one image and bounding box in in a subplot to show the user the class
        classes_below_tolerance = class_counts[class_counts <= self.unbalanced_class_tolerance].index
        if len(classes_below_tolerance) > 0 :
            fig, axes = plt.subplots(self.unbalanced_class_tolerance + 1, len(classes_below_tolerance), figsize=(30, 30))
            fig.suptitle("Classes below tolerance")
            for i,class_name in enumerate(classes_below_tolerance):
                class_images = self.df_dt[self.df_dt['object-class'] == class_name].to_dict(orient='records')
                for j,elmt in enumerate(class_images) :
                    im_path = os.path.join(self.raw_images_path, elmt['imagename'])
                    image = cv2.imread(im_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    x_center, y_center, width, height = int(float(elmt['x_center'])), int(float(elmt['y_center'])), int(float(elmt['width'])), int(float(elmt['height']))

                    cv2.rectangle(image, (x_center - width//2, y_center - height//2), (x_center + width//2, y_center + height//2), (0, 255, 0), 2)
                    cv2.putText(image, class_name, (x_center - width//2, y_center - height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    axes[j][i].imshow(image)
                    axes[j,i].axis('off')

                for j in range(0, self.unbalanced_class_tolerance + 1) :
                    axes[j,i].axis('off')
                
                cropped_img = image[y_center - height//2:y_center + height//2, x_center - width//2:x_center + width//2]
                axes[self.unbalanced_class_tolerance][i].imshow(cropped_img)
                axes[0][i].set_title(f'Class {class_name}')
            if display == True :
                plt.show()
        
        plt.savefig(os.path.join(self.info_path, "classes_under_represented.png").replace("\\", "/"))
        plt.close()
            

        # Plot the repartition of the size of the bounding boxes along the classes
        df = self.df_dt.copy()
        df['area'] = df['width'] * df['height']
        class_sizes = df.groupby('object-class')['area'].mean()
        class_sizes = class_sizes.sort_values()
        # Create a bar chart to visualize the distribution of box sizes for each class
        plt.figure(figsize=(12, 6))
        plt.bar(class_sizes.index, class_sizes.values)
        plt.title("Distribution of Bounding Box Sizes Along Classes")
        plt.xlabel("Classes")
        plt.ylabel("Average Box Size")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        if display == True :
            plt.show()

        plt.savefig(os.path.join(self.info_path, "size_repartition.png").replace("\\", "/"))

        small_objects = class_sizes.head(10).to_dict()

        f = open(os.path.join(self.info_path, "problematic_summary.txt").replace("\\", "/"), "w")
        for id in small_objects :
            if id in classes_below_tolerance :
                f.write("{} is under-represented and annotations part of the smallest object size of the dataset".format(id))
        f.close()

    def __create_class_uniform_train_val_test(self) : 
        # Extract from self.df_dt the images name and the corresponding height in a list
        images_classes = self.df_dt.groupby("imagename")["object-class"].apply(list).to_dict()
        for im in self.empty_images :
            images_classes[im] = ["-1"]

        class_distribution = {}
        for k,v in images_classes.items() :
            for c in v :
                if c in class_distribution :
                    class_distribution[c] += 1
                else :
                    class_distribution[c] = 1

        train_class_distribution = {}
        val_class_distribution = {}
        test_class_distribution = {}
        for k,v in class_distribution.items() :
            if int(v * self.ratio_train) < 1 or int(v * self.ratio_val) < 1 or int(v * self.ratio_test) < 1 :
                if v >= 3 :
                    train_class_distribution[k] = 1
                    val_class_distribution[k] = 1
                    test_class_distribution[k] = v-2
                    print(f"Class {k} : train : 1 / val : 1 / test : {v-2}")
                else :
                    print(f"Class {k} has too few images to be split : {v}")
            else :

                train_class_distribution[k] = int(v * self.ratio_train)
                val_class_distribution[k]   = int(v * self.ratio_val)
                test_class_distribution[k]  = int(v * self.ratio_test)
                if k == "-1" :
                    train_class_distribution[k] = int(v//2 * self.ratio_train)
                    val_class_distribution[k] = int(v//2 * self.ratio_val)
                    test_class_distribution[k] = int(v//2 * self.ratio_test)

        # Create uniform sets       
        train,test, train_class_distribution, test_class_distribution, images_ = self.__split(images_classes,train_class_distribution,test_class_distribution)
        val,test, val_class_distribution, test_class_distribution, images_  = self.__split(images_,val_class_distribution,test_class_distribution)

        print("test")
        print(test_class_distribution)
        for c in test_class_distribution :
            done = False
            counter = 0
            if test_class_distribution[c] >= 0:
                print(c, test_class_distribution[c])
                for im in train :
                    if c in images_classes[im] and not done :
                        counter +=1 
                    if counter >= 2 and not done :
                        train.remove(im)
                        test.append(im)
                        test_class_distribution[c] = -1
                        done = True

        print("val")
        print(val_class_distribution)
        for c in val_class_distribution :
            done = False
            counter = 0
            if val_class_distribution[c] >= 0  :
                print(c, val_class_distribution[c])
                for im in train :
                    if c in images_classes[im] and not done :
                        counter +=1 
                    if counter >= 2 and not done :
                        train.remove(im)
                        val.append(im)
                        val_class_distribution[c] = -1
                        done = True
        
        print("train")
        train_class_distribution['0'] = 1
        print(train_class_distribution)
        for c in train_class_distribution :
            done = False
            counter = 0
            if train_class_distribution[c] >= 0  :
                print(c, train_class_distribution[c])
                for im in test :
                    if c in images_classes[im] and not done :
                        counter +=1 
                    if counter >= 2 and not done :
                        test.remove(im)
                        train.append(im)
                        train_class_distribution[c] = -1
                        done = True
            
        # Adjusting , TODO : rework this splits methods
        for c in train_class_distribution :
            if c in ["24","26","32","34","6","27","43","4","14"] :
                print(c)
                done = False
                for im in train :
                    if c in images_classes[im] :
                        if done == False :
                            train.remove(im)
                            test.append(im)
                            done = True

        for c in train_class_distribution :
            if c in ["43"] :
                print(c)
                done = False
                for im in train :
                    if c in images_classes[im] :
                        if done == False :
                            train.remove(im)
                            test.append(im)
                            done = True


        # verify that an image in a is not in the other
        for im in train :
            if im in val or im in test :
                raise Exception("An image is in train and val or test")
            if im in self.empty_images :
                print("An empty image is in train")
        for im in val :
            if im in train or im in test :
                raise Exception("An image is in val and train or test")
            if im in self.empty_images :
                print("An empty image is in val")
        for im in test :
            if im in train or im in val :
                raise Exception("An image is in test and train or val") 
            if im in self.empty_images :
                print("An empty image is in test")
        
        print("Done splitting : ")
        print(len(images_classes), len(train), len(val), len(test), sum([len(train), len(val), len(test)]))   
        # self.__plot_class_repartition(train,val,test)
        k = input("Press [Enter] : continue | [q] cancel |")
        if k == "q" :
            exit()
        self.__move_images_and_labels(train, val, test)
        self.__plot_class_repartition(train,val,test)

    def __split(self, images_classes,train_class_distribution,test_class_distribution) :
        train = []
        test = []
        im_to_pop = []
        images_classes = images_classes.copy()
        train_class_distribution = train_class_distribution.copy()
        test_class_distribution = test_class_distribution.copy()
        for im,classes in images_classes.items() :
            add_to_train_set = False
            for c in classes :
                if train_class_distribution[c] > 0 :
                    add_to_train_set = True
            
            if add_to_train_set :
                for c in classes :
                    train_class_distribution[c] -= 1
                train.append(im)
                im_to_pop.append(im)

            else :
                for c in classes :
                    test_class_distribution[c] -= 1
                test.append(im)

        for im in im_to_pop :
            images_classes.pop(im)

        return train, test, train_class_distribution, test_class_distribution, images_classes

    def __create_fld_architecture(self) :
        # create the dataset folder
        parent_folder = os.path.join(self.root, self.name)
        os.mkdir(parent_folder)
        # create the train folder
        os.mkdir(os.path.join(parent_folder, "train"))
        os.mkdir(os.path.join(parent_folder, "train", "images").replace("\\", "/"))
        os.mkdir(os.path.join(parent_folder, "train", "labels").replace("\\", "/"))
        # create the val folder
        os.mkdir(os.path.join(parent_folder, "val"))
        os.mkdir(os.path.join(parent_folder, "val", "images").replace("\\", "/"))
        os.mkdir(os.path.join(parent_folder, "val", "labels").replace("\\", "/"))
        # create the test folder
        os.mkdir(os.path.join(parent_folder, "test"))
        os.mkdir(os.path.join(parent_folder, "test", "images").replace("\\", "/"))
        os.mkdir(os.path.join(parent_folder, "test", "labels").replace("\\", "/"))
        # create info folder
        os.mkdir(os.path.join(parent_folder, "info"))

    # Move images and labels using shutils
    # Add counter checkup to make sure that the number of images and labels in the train, val and test sets are correct
    def __move_images_and_labels(self, train, val, test) :
        # move images and labels to the train set
        total = len(train) + len(val) + len(test)
        train_c = 0
        val_c = 0
        test_c = 0

        train_fp = open(os.path.join(self.info_path, "train_images.txt").replace("\\", "/"), "w")
        val_fp = open(os.path.join(self.info_path, "val_images.txt").replace("\\", "/"), "w")
        test_fp = open(os.path.join(self.info_path, "test_images.txt").replace("\\", "/"), "w")

        for i,im in enumerate(train) :
            shutil.copy(os.path.join(self.raw_images_path, im), os.path.join(self.train_images_path, im).replace("\\", "/"))
            shutil.copy(os.path.join(self.raw_labels_path, im.split(".")[0] + self.labels_extension_ftype), os.path.join(self.train_labels_path, im.split(".")[0] + self.labels_extension_ftype).replace("\\", "/"))
            train_c += 1
            train_fp.write(im + "\n")
            print(f"Moving images and labels to the train set : {round(train_c/total*100,2)} %", end="\r")
        # move images and labels to the val set
        for i,im in enumerate(val) :
            shutil.copy(os.path.join(self.raw_images_path, im), os.path.join(self.val_images_path, im).replace("\\", "/"))
            shutil.copy(os.path.join(self.raw_labels_path, im.split(".")[0] + self.labels_extension_ftype), os.path.join(self.val_labels_path, im.split(".")[0] + self.labels_extension_ftype).replace("\\", "/"))
            val_c += 1
            val_fp.write(im + "\n")
            print(f"Moving images and labels to the val set : {round(val_c/total*100,2)} %", end="\r")
        # move images and labels to the test set
        for i,im in enumerate(test) :
            shutil.copy(os.path.join(self.raw_images_path, im), os.path.join(self.test_images_path, im).replace("\\", "/"))
            shutil.copy(os.path.join(self.raw_labels_path, im.split(".")[0] + self.labels_extension_ftype), os.path.join(self.test_labels_path, im.split(".")[0] + self.labels_extension_ftype).replace("\\", "/"))
            test_c += 1
            test_fp.write(im + "\n")
            print(f"Moving images and labels to the test set : {round(test_c/total*100,2)} %", end="\r")

        if total != train_c + val_c + test_c :
            raise Exception("Number of images and labels in the train, val and test sets are incorrect")
        
        self.number_of_images = total
        self.number_of_test_images = test_c
        self.number_of_train_images = train_c
        self.number_of_val_images = val_c
        print("Images and labels moved to the train, val and test sets")
        
    # this function plots the class repartition is the train, val and test sets folders
    # 3 subplots : one for each set and use barh to plot the repartition of the classes
    # args : train val and test are the list of images in the train, val and test sets
    def __plot_class_repartition(self, train, val, test, show=False) :
        train_classes = []
        val_classes = []
        test_classes = []
        for im in train :
            lb = im.split(".")[0] + self.labels_extension_ftype
            lb_path = os.path.join(self.raw_labels_path, lb)
            if not os.path.getsize(lb_path) > 0 :
                train_classes.append("empty")
            with open(lb_path, "r") as f :
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                train_classes.extend([line.split(" ")[0] for line in lines])
        for im in val :
            lb = im.split(".")[0] + self.labels_extension_ftype
            lb_path = os.path.join(self.raw_labels_path, lb)
            if not os.path.getsize(lb_path) > 0 :
                val_classes.append("empty")
            with open(lb_path, "r") as f :
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                val_classes.extend([line.split(" ")[0] for line in lines])
        for im in test :
            lb = im.split(".")[0] + self.labels_extension_ftype
            lb_path = os.path.join(self.raw_labels_path, lb)
            if not os.path.getsize(lb_path) > 0 :
                test_classes.append("empty")
            with open(lb_path, "r") as f :
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                test_classes.extend([line.split(" ")[0] for line in lines])
        
        train_classes = pd.Series(train_classes).value_counts().sort_index()
        val_classes = pd.Series(val_classes).value_counts().sort_index()
        test_classes = pd.Series(test_classes).value_counts().sort_index()
        classes = pd.concat([train_classes, val_classes, test_classes], axis=1)
        ax = classes.plot(kind='barh', stacked=True, figsize=(10, 6))
        plt.xlabel('Counts')
        plt.ylabel('Label')
        plt.title('Cumulative repartition of classes along sets')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(['train','val','test'], title='Split Type', loc='upper right')
        if show :
            plt.show()
        plt.savefig(os.path.join(self.info_path, "dataset_repartition.png").replace("\\", "/"))
        
    # Set param function loads a yaml file containing the parameters of the dataset :
    # - the name of the dataset
    # - the ratio of images in the train set
    # - the ratio of images in the val set
    # - the ratio of images in the test set
    # - the classes names file path
    # - the folder path containing the images
    # - the folder path containing the labels
    def get_params(self, yaml_path) :
        with open(yaml_path, "r") as f :
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.name = params["name"]
            self.ratio_train = params["ratio_train"]
            self.ratio_val   = params["ratio_val"]
            self.ratio_test  = params["ratio_test"]
            self.classes_names_path = params["classes_names_path"]
            self.raw_images_path = params["raw_images_path"]
            self.raw_labels_path = params["raw_labels_path"]
        print("Params loaded")
