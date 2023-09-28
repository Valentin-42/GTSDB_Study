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

    label_dir = "./datasets/GTSDB/train/labels/"
    image_dir = "./datasets/GTSDB/train/images/"

    # plot image 4 by 4 
    fig, ax = plt.subplots(2,2)
    j,k = 0,0
    for i,img in enumerate(os.listdir(image_dir)) :
        print(j,k)
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



if __name__ == "__main__":
    main()
    # show()