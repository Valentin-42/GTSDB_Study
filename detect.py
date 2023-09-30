import YOLO
import os

def detect(weights) :
    
    model = YOLO(weights)
    model('./GTSDB/test/images/', save=True)

    print("Done!")
