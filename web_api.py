import flask
from flask_cors import CORS
from flask import request, jsonify
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import requests
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def prepare_pridctor():

    classes = ['abutilon_julianae_leaf', 'abutilon_julianae_flower', 'acacia_anomala_flower', 'acacia_anomala_leaf', 'acacia_constablei_flower', 'acacia_constablei_leaf', 'boronia_umbellata_flower', 'boronia_umbellata_leaf']

    data_path = '/labelme_image_detection/'

    for d in ["train", "val"]:
      DatasetCatalog.register("eps_" + d, lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("eps_" + d).set(thing_classes=classes)

    microcontroller_metadata = MetadataCatalog.get("eps_train")
    
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("eps_train",)
    cfg.DATASETS.TEST = ("eps_val")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    
    classes = MetadataCatalog.get("eps_val").thing_classes
    predictor = DefaultPredictor(cfg)
    
    return (predictor, classes)

def perform_prediction(predictor: DefaultPredictor, img_path: str):
    
    pred_list= list()
    img = cv2.imread(img_path)
    outputs = predictor(img)
    ins = outputs['instances']
    
    for i in range(len(ins)):
      pred_dict = dict()
      pred_class = classes[ins.pred_classes[i].item()]
      score = ins.scores[i].item()
      
      # check for score is more than 50%
      if score > 0.50:
        pred_dict["Scientific_Name"] = pred_class.replace('_', ' ').replace('flower', '').replace('leaf', '').strip()
        pred_dict["score"] = "{:.2f}".format(score*100)
        pred_list.append(pred_dict)
    
    print(pred_list)  
    
    return pred_list
    

app = flask.Flask(__name__)
CORS(app)
predictor, classes = prepare_pridctor()


@app.route("/predictPlant", methods = ['GET', 'POST'])
def predict_plant():
    if request.method == 'GET':
        return "Server dp is running on port 80."
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "images/" + img.filename
        img.save(img_path)
        
        pred_list = perform_prediction(predictor, img_path)
    
        return jsonify(pred_list)

print("Server dp is running on port 80.")

app.run(host="0.0.0.0", port=80)
