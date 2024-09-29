from utils_rtdetr.rtdetr import load_model
from torchvision import transforms
import cv2
import numpy as np
import torch
import argparse
import os
from deep_utils import split_extension

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', default='rtdetr_resnet50')
parser.add_argument('--input', default='../images/rtdetr_test/')
parser.add_argument('--output', default='../outputs/cam_test/rtdetr_test')
parser.add_argument('--device', help='cpu or cuda', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
args = parser.parse_args()



def draw_boxes(outputs, orig_image):
    np.random.seed(42)
    mscoco_category2name = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }
    mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
    mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
    COLORS = np.random.uniform(0, 255, size=(len(mscoco_category2name), 3))

    for i in range(len(outputs['pred_boxes'][0])): # loop per object box (total 300)
        logits = outputs['pred_logits'][0][i] # extraxt 80 values of each object
        # print('Logit: ', logits)
        soft_logits = torch.softmax(logits, dim=-1) # probability
        max_index = torch.argmax(soft_logits).cpu() # max confidence out of the 80 values
        class_name = mscoco_category2name[mscoco_label2category[int(max_index.numpy())]]

        if soft_logits[max_index] > 0.50: # filters out low probability detections out of 300 detections
            box = outputs['pred_boxes'][0][i].cpu().numpy()
            cx, cy, w, h = box
            cx = cx * orig_image.shape[1]
            cy = cy * orig_image.shape[0]
            w = w * orig_image.shape[1]
            h = h * orig_image.shape[0]

            x1 = int(cx - (w // 2))
            y1 = int(cy - (h // 2))
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            color = COLORS[max_index]

            confidence = soft_logits[max_index].numpy().tolist()
            confidence = round(confidence, 2)  # or "{:.2f}".format(confidence)
            label = '{0}: {1:.2f}'.format(class_name, confidence)

            cv2.rectangle(
                orig_image,
                (x1, y1),
                (x2, y2),
                thickness=2,
                color=color,
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                orig_image,
                text=label,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                thickness=2,
                color=color,
                lineType=cv2.LINE_AA
            )

    return orig_image

def manage_score(outputs, orig_image):
    np.random.seed(42)
    mscoco_category2name = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }
    mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
    mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
    COLORS = np.random.uniform(0, 255, size=(len(mscoco_category2name), 3))

    logits = []
    boxes = []
    cls_names = []
    cls_ids = []
    confidences = []

    for i in range(len(outputs['pred_boxes'][0])):  # loop per object box (total 300)
        logit = outputs['pred_logits'][0][i]  # extraxt 80 values of each object
        # print('Logit: ', type(logit))
        soft_logit = torch.softmax(logit, dim=-1)  # probability
        max_index = torch.argmax(soft_logit).cpu()  # max confidence out of the 80 values
        class_name = mscoco_category2name[mscoco_label2category[int(max_index)]]

        if soft_logit[max_index] > 0.50:  # filters out low probability detections out of 300 detections
            logits.append(logit)

            cx, cy, w, h = outputs['pred_boxes'][0][i].cpu().numpy().tolist()
            print(f"Orig_image: {orig_image.shape}")
            cx = cx * orig_image.shape[1]
            cy = cy * orig_image.shape[0]
            w = w * orig_image.shape[1]
            h = h * orig_image.shape[0]
            x1 = int(cx - (w // 2))
            y1 = int(cy - (h // 2))
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            box = [x1, y1, x2, y2]

            cls_name = class_name
            cls_id = mscoco_category2label[mscoco_label2category[int(max_index)]]
            confidence = soft_logit[max_index].numpy().tolist()
            confidence = round(confidence, 2) # or "{:.2f}".format(confidence)

            boxes.append(box)
            cls_names.append(cls_name)
            cls_ids.append(cls_id)
            confidences.append(confidence)

            color = COLORS[max_index]

            cv2.rectangle(
                orig_image,
                (x1, y1),
                (x2, y2),
                thickness=2,
                color=color,
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                orig_image,
                text=class_name,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                thickness=2,
                color=color,
                lineType=cv2.LINE_AA
            )

    preds = [[boxes], [cls_ids], [cls_names], [confidences]]
    logits_tensor = torch.stack(logits)
    return preds, logits_tensor, orig_image

def run_rtdetr():
    # Load model.
    # Choose model between,
    # rtdetr_resnet18, rtdetr_resnet34, rtdetr_resnet50, rtdetr_resnet101
    model = load_model(args.model)
    model.eval().to(args.device)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    input_images = [f for f in os.listdir(args.input) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through each image in the folder
    for image_name in input_images:
        # Load image
        image_path = os.path.join(args.input, image_name)
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])
        image = transform(image)
        image = image.unsqueeze(0).to(args.device)

        # Forward pass.
        with torch.no_grad():
            outputs = model(image)
            # outputs['pred_logits'] has a shape of [b, 300, 80], i.e., [batch, 300, coco class]
            # outputs['pred_boxes'] has a shape of [b, 300, 4], i.e., [batch, 300, bbox coordinates]

        print(f"Outputs: {outputs['pred_logits'].shape}\n{outputs['pred_boxes'].shape}")

        # preds, logits, orig_image = manage_score(outputs, orig_image)
        # print(f"Preds {len(preds)}: {preds}")
        # print(f"Logits {logits.shape}: {logits[0]}")
        orig_image = draw_boxes(outputs, orig_image)
        # cv2.imshow('Image', orig_image)
        # cv2.waitKey(0)
        # # file_name = args.input.split(os.path.sep)[-1]
        file_name = split_extension(image_name, suffix='_org')
        output_path = os.path.join(args.output, file_name)
        cv2.imwrite(output_path, orig_image)

if '__name__' == '__main__':
    run_rtdetr()
else:
    run_rtdetr()