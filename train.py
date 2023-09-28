from ultralytics import YOLO
import yaml
import argparse

def optimizer_tuning(path_to_weights, path_to_config) :
    model = YOLO(path_to_weights)

    # Default params
    epochs   = 200
    imgsz    = 640
    save_period = 10
    data     = path_to_config
    device   = 0
    exist_ok = True
    batch    = 128
    project_name = "Optimizer_Tuning" 
    # 
    experiments = {
            'Opt-SGD': {'optimizer':'SGD', 'lr0':0.01, 'lrf':0.01},
            'Opt-Adam': {'optimizer':'Adam', 'lr0':0.01, 'lrf':0.01},
            'Opt-SGD2': {'optimizer':'SGD', 'lr0':0.02, 'lrf':0.01},
            'Opt-Adam2': {'optimizer':'Adam', 'lr0':0.02, 'lrf':0.01},
            'Opt-SGD5': {'optimizer':'SGD', 'lr0':0.05, 'lrf':0.01},
            'Opt-Adam5': {'optimizer':'Adam', 'lr0':0.05, 'lrf':0.01},
        }

    for exp in experiments.keys() : 
        name = exp
        optimizer = experiments[exp]["optimizer"]
        lr0 = experiments[exp]["lr0"]
        lrf = experiments[exp]["lrf"]

        results = model.train(
            data = data,
            epochs = epochs,
            imgsz = imgsz,
            save_period = save_period,
            device = device,
            exist_ok = exist_ok,
            batch = batch,
            project = project_name,
            name = name,
            patience = epochs,
            optimizer = optimizer,
            lr0 = lr0,
            lrf = lrf
        )

def hpp_tuning(path_to_weights, path_to_config, epochs) :
    # Default params
    epochs   = epochs
    imgsz    = 640
    patience = epochs
    save_period = 10
    data     = path_to_config
    device   = 0
    exist_ok = True
    batch    = 128
    project_name = "Hyperparams_Tuning" 
    optimizer = 'SGD'

    curve = [0.0, 0.35, 0.65, 1.0]
    i,j,k,l = 0,0,0,0
    for exp in range(256) : 
        model = YOLO(path_to_weights)

        name = f"exp_{exp}_{i}_{j}_{k}_{l}"
        mosaic = curve[i] 
        mixup = curve[j]
        copy_paste = curve[k]
        scale = curve[l]

        results = model.train(
            data = data,
            epochs = epochs,
            patience = patience,
            imgsz = imgsz,
            save_period = save_period,
            device = device,
            exist_ok = exist_ok,
            batch = batch,
            project = project_name,
            name = name,
            optimizer = optimizer,
            mosaic = mosaic,
            mixup = mixup,
            copy_paste = copy_paste,
            scale = scale,
            resume = False
        )
        i+= 1
        if i == 3 :
            i=0
            j+=1
        elif j ==3:
            j=0
            k+=1
        elif k==3:
            k=0
            l+=1
        elif l==3 :
            l=0
            break


def train(path_to_weights, path_to_config,epochs, augment) :

    model = YOLO(path_to_weights)

    # Default params
    epochs   = int(epochs)
    imgsz    = (1360,800)
    save_period = 100
    data     = path_to_config
    device   = 0
    exist_ok = True
    cache = True
    batch = 128
    project_name = "GTSDB_Training"
    name = 'first_training'
    optimizer = 'SGD'
    lr0 =float(0.01)
    lrf =float(0.01)
    box = 7.5
    cls = 0.5

    hsv_h= 0  # (float) image HSV-Hue augmentation (fraction)
    hsv_s= 0  # (float) image HSV-Saturation augmentation (fraction)
    hsv_v= 0  # (float) image HSV-Value augmentation (fraction)
    degrees= 0 # (float) image rotation (+/- deg)
    translate= 0  # (float) image translation (+/- fraction)
    scale= 0  # (float) image scale (+/- gain)
    shear= 0  # (float) image shear (+/- deg)
    perspective= 0.0  # (float) image perspective (+/- fraction), range 0-0.001
    flipud= 0.0  # (float) image flip up-down (probability)
    fliplr= 0  # (float) image flip left-right (probability)
    mosaic= 0  # (float) image mosaic (probability)
    mixup= 0.0  # (float) image mixup (probability)
    copy_paste= 0.0  # (float) segment copy-paste (probability)


    resume = False
    # 
    print("Starting Training ...")

    results = model.train(
        data = data,
        epochs = epochs,
        imgsz = imgsz,
        save_period = save_period,
        device = device,
        exist_ok = exist_ok,
        batch = batch,
        project = project_name,
        name = name,
        cache = cache,
        patience = epochs,
        optimizer = optimizer,
        lr0 = lr0,
        lrf = lrf,
        augment=augment,
        mosaic = mosaic,
        mixup = mixup,
        hsv_h = hsv_h,
        hsv_s = hsv_s,
        hsv_v = hsv_v,
        degrees = degrees,
        translate = translate,
        shear = shear,
        perspective = perspective,
        flipud = flipud,
        fliplr = fliplr,
        copy_paste = copy_paste,
        scale = scale,
        box = box,
        resume = resume
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", default="yolov8n.pt", help = "path to .pt")
    parser.add_argument("-c", default="./param/", help = "data file path")
    parser.add_argument("-e", default=100, help = "nbs of epochs")
    parser.add_argument("-a", default=True, help = "apply augment")


    args = parser.parse_args()
    # optimizer_tuning(args.w, args.c)
    # hpp_tuning(args.w, args.c, args.e)
    train(args.w, args.c,args.e, args.a)