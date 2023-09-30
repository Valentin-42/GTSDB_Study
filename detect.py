from ultralytics import YOLO
import os
import argparse
import pandas as pd

def detect(weights) :
    
    model = YOLO(weights)
    model('./datasets/GTSDB/test/images/', save=True, save_txt=True, save_conf=True)

    print("Done!")


def detect_v2(weights) :
    
    model = YOLO(weights)
    metrics = model.val(split="test")

    results_dict = metrics.results_dict
    speed = metrics.speed
    ap_class_index = metrics.ap_class_index

    df = pd.DataFrame(results_dict, index=[0])
    df["inference time"] = speed["inference"]
    df["preprocess time"] = speed["preprocess"]
    df["postprocess time"] = speed["postprocess"]
    df.to_csv("./results.csv", index=False)

    print("Results!")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    detect_v2(parser.parse_args().w)