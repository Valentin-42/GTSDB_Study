from ultralytics import YOLO
import os
import argparse

def detect(weights) :
    
    model = YOLO(weights)
    model('./datasets/GTSDB/test/images/', save=True, save_txt=True, save_conf=True)

    print("Done!")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    detect(parser.parse_args().w)