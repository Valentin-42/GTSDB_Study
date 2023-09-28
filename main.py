from DS import Dataset as DS
import GTSDB_toolbox
import yaml


def main():
    print("main.py")
    working_dir = "./datasets/full/raw/"
    # load yaml file 
    with open("./params/repartitions.yml", "r") as f :
        mapping = yaml.load(f, Loader=yaml.FullLoader)

    GTSDB_toolbox.ppm_to_png(working_dir)
    GTSDB_toolbox.format_annotation_GTSDB_to_YOLOv3(working_dir + "gt.txt", save=True, merge_classes=False, merge_classes_dict=mapping["3_classes"], normalize=True)
    dataset = DS("./datasets/","GTSDB",True,"./params/GTSDB_params.yml")
    

    
def show() :
    import os 
    import cv2
    import matplotlib.pyplot as plt

    label_dir = "./datasets/full/raw/labels/"
    image_dir = "./datasets/full/raw/png/"

    for img in os.listdir(image_dir) :
        print(image_dir + img)
        GTSDB_toolbox.visualize(image_dir + img, label_dir + img.split(".")[0] + ".txt")
        k =input("Press Enter to continue...")
        if k == "q" :
            break

if __name__ == "__main__":
    main()
    # show()