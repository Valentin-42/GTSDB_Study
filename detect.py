from ultralytics import YOLO
import os
import argparse

def detect(weights) :
    
    model = YOLO(weights)
    model('./GTSDB/test/images/', save=True)

    print("Done!")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    detect(parser.parse_args().w)