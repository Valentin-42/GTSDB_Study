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

if __name__ == "__main__":
    main()