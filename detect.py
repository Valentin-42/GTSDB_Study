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

    # metrics is an object metrics with attributes:
    # ap_class_index: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 35, 36, 38, 39, 40, 41, 42])
    # box: ultralytics.utils.metrics.Metric object
    # confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x7efd66f32070>
    # fitness: 0.33166380485783953
    # keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    # maps: array([          0,     0.76514,       0.528,      0.0705,     0.28983,     0.30446,     0.48391,     0.21896,     0.45817,     0.45879,     0.88657,     0.31266,     0.77495,     0.81895,     0.55557,     0.53675,           0,     0.56988,     0.30506,     0.13267,      0.0728,    0.072364,     0.21695,    0.032057,
    #            0.26092,      0.2879,     0.14404,     0.32663,     0.27445,    0.040705,     0.46212,     0.32663,       0.398,     0.39219,     0.32663,    0.073621,    0.062838,     0.32663,     0.81694,    0.031982,     0.22727,           0,     0.40059])
    # names: {0: 'speed limit 20', 1: 'speed limit 30', 2: 'speed limit 50', 3: 'speed limit 60', 4: 'speed limit 70', 5: 'speed limit 80', 6: 'restriction ends 80', 7: 'speed limit 100', 8: 'speed limit 120', 9: 'no overtaking', 10: 'no overtaking', 11: 'priority at next intersection', 12: 'priority road', 13: 'give way', 14: 'stop', 15: 'no traffic both ways', 16: 'no trucks', 17: 'no entry', 18: 'danger', 19: 'bend left', 20: 'bend right', 21: 'bend', 22: 'uneven road', 23: 'slippery road', 24: 'road narrows', 25: 'construction', 26: 'traffic signal', 27: 'pedestrian crossing', 28: 'school crossing', 29: 'cycles crossing', 30: 'snow', 31: 'animals', 32: 'restriction ends', 33: 'go right', 34: 'go left', 35: 'go straight', 36: 'go right or straight', 37: 'go left or straight', 38: 'keep right', 39: 'keep left', 40: 'roundabout', 41: 'restriction ends', 42: 'restriction ends'}
    # plot: True
    # results_dict: {'metrics/precision(B)': 0.39369254630595557, 'metrics/recall(B)': 0.3605528202699014, 'metrics/mAP50(B)': 0.3769708241485492, 'metrics/mAP50-95(B)': 0.32662969160331623, 'fitness': 0.33166380485783953}
    # save_dir: PosixPath('runs/detect/val')
    # speed: {'preprocess': 0.9860304378023085, 'inference': 2.681037723618066, 'loss': 0.0005920461360240143, 'postprocess': 0.35007688023099964}
    # Extract maps, results dict, and speed from metrics object
    # store it in a dataframe pandas
    results_dict = metrics.results_dict
    speed = metrics.speed
    ap_class_index = metrics.ap_class_index

    df = pd.DataFrame(results_dict, index=[0])
    df["inference time"] = speed["inference"]
    df["preprocess time"] = speed["preprocess"]
    df["postprocess time"] = speed["postprocess"]
    df.to_csv("./results.csv", index=False)

    print("Results!")


    df_class = pd.DataFrame(ap_class_index, columns=["class_index"])
    df_class["class_name"] = [metrics.names[i] for i in ap_class_index]
    maps = []
    for i in ap_class_index :
        maps.append(metrics.class_result(i))

    df_class["mAP"] = maps
    df_class.to_csv("./results_class.csv", index=False)

    print("Results_per_class!")



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    detect_v2(parser.parse_args().w)