from ultralytics import YOLO
import os
import argparse

def detect(weights) :
    
    model = YOLO(weights)
    model('./datasets/GTSDB/test/images/', save=True, save_txt=True, save_conf=True)

    print("Done!")


def detect_v2(weights) :
    
    model = YOLO(weights)
    metrics = model.val(split="test")

    print("Done!")
    print(metrics)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    detect_v2(parser.parse_args().w)