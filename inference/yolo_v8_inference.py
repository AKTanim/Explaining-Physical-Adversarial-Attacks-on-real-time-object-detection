"""
YOLO v8 inference.
"""
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from deep_utils import Box


def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, cls_name, (x1, y1))
    return res_img

def process_score(results):
    bbox = results[0].boxes.xyxy
    conf = results[0].boxes.conf
    cls_id = []
    cls_names = []

    for id in results[0].boxes.cls.int().tolist():
        cls_id.append(id)

        name = results[0].names[id]
        cls_names.append(name)

    cls_id = torch.tensor(cls_id)
    print(f"Boxes: {bbox}\nConfidence: {conf}\nClass ids: {cls_id}\nClass names{type(cls_names)}: {cls_names}")
    return bbox, cls_names, cls_id, conf

def load_image(img_path):
    # To do: list of images
    img = cv2.imread(img_path)
    cv2.imshow('Input image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def run_yolov8():
    model = YOLO("yolov8n.pt") # Download the model beforehand

    # Load and show input image
    img_path = 'images/inria(5).png'
    #img = load_image(img_path)

    # inference and process results
    results = model.predict(img_path, conf=0.25)
    print(f"Forward:\n{results[0]}")
    results[0].show()
    bbox, cls_name, cls_id, conf = process_score(results)

    return model, bbox, cls_name, cls_id, conf


if __name__ == '__main__':
    run_yolov8()