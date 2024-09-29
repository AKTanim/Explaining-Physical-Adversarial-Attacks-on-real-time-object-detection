from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from inference.utils_yolov2.util_grad import *
import argparse
import os 
import os.path as osp
from inference.utils_yolov2.darknet import Darknet
from inference.utils_yolov2.preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
from utils.general import xywh2xyxy
from utils.metrics import box_iou
import torchvision

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)

class YOLOv2OD(nn.Module):
    def __init__(self,
                 dataset,
                 device,
                 img_size,
                 names=None,
                 mode='eval',
                 confidence=0.5,
                 nms=0.4,
                 batch_size=1):
        super(YOLOv2OD, self).__init__()
        self.device = device
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.nms = nms
        self.batch_size = batch_size

        if dataset == "pascal":
            self.inp_dim = 416
            self.num_classes = 20
            self.classes = load_classes('inference/utils_yolov2/data/voc.names')
            weightsfile = 'inference/utils_yolov2/weights/yolo-voc.weights'
            cfgfile = "inference/utils_yolov2/cfg/yolo-voc.cfg"


        elif dataset == "coco":
            self.inp_dim = 544
            self.num_classes = 80
            self.classes = load_classes('inference/utils_yolov2/data/coco.names')
            weightsfile = 'inference/utils_yolov2/weights/yolov2.weights'
            cfgfile = "inference/utils_yolov2/cfg/yolo.cfg"

        else:
            print("Invalid dataset")
            exit()

        self.stride = 32

        # Set up the neural network
        print("Loading network.....")
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        print("Network successfully loaded")

        # If there's a GPU availible, put the model on GPU
        self.model.requires_grad_(True)
        self.model.to(self.device)

        print("==================================================================================")
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                if param.dim() == 1:
                    print(f"{name:<40}{param.size()}\t\t{num_params}")
                else:
                    print(f"{name:<40}{list(param.size())}\t\t\t{num_params}")

        print("==================================================================================")
        print(self.model)

        if self.mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        # print("Target layer:\n",self.model.module_list._modules['29'])
        # fetch the names
        if names is None:
            print('[INFO] fetching names from coco file')
            self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow',
                          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                          'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                          'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                          'apple',
                          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                          'couch',
                          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                          'keyboard', 'cell phone',
                          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                          'teddy bear',
                          'hair drier', 'toothbrush']
        else:
            self.names = names

        # preventing cold start
        # img = torch.zeros((1, 3, *self.img_size), device=device)
        # self.model(img)
        self.loss_target = lambda obj, cls: obj * cls  # for yolo only


    def get_test_input(self, input_dim):
        img = cv2.imread("inference/utils_yolov2/imgs/dog.jpg")
        img = cv2.resize(img, (input_dim, input_dim))
        img_ =  img[:,:,::-1].transpose((2,0,1))
        img_ = img_[np.newaxis,:,:,:]/255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)

        img_ = img_.to(self.device)
        # self.num_classes
        return img_


    def non_max_suppression(self, prediction, logits, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference and logits results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]
        for xi, (x, log_) in enumerate(zip(prediction, logits)):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            log_ = log_[xc[xi]]
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # log_ *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                # log_ = x[:, 5:]
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                log_ = log_[conf.view(-1) > conf_thres]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            logits_output[xi] = log_[i]
            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output, logits_output

    def forward(self, img):
        # self.model(self.get_test_input(self.inp_dim))

        # img, orig_img, img_dim = prep_image(img, self.inp_dim)
        # print('imageshp...', img.shape)
        score = self.model(img)
        output = score.view(score.shape[0], 5*85, score.shape[2]*score.shape[3])
        output = output.transpose(1,2).contiguous()
        output = output.view(score.shape[0], score.shape[2]*score.shape[3]*5, 85)
        print('output: ', output.shape)
        preds, logits = self.non_max_suppression(output, output[:,:,5:])
        anchors = self.model.anchors
        return score.data, anchors, logits

        # prediction = output
        # prediction = predict_transform(score, self.inp_dim, self.stride, self.model.anchors,
        #                                self.num_classes, self.confidence, CUDA=torch.cuda.is_available())
        # print('logits_2:', prediction.shape)
        #
        # # perform NMS on these boxes, and save the results
        # # I could have done NMS and saving seperately to have a better abstraction
        # # But both these operations require looping, hence
        # # clubbing these ops in one loop instead of two.
        # # loops are slower than vectorised operations.
        #
        # prediction, _ = self.write_results(prediction, self.num_classes, nms=True, nms_conf=self.nms)
        # print('logits_3: ', prediction.shape)
        # # print("logit_grad:", logits.requires_grad)
        # prediction = prediction.view(-1, 8)
        # logits = [logits]
        # # normalized_logits = (logits - logits.min())*100 / (logits.max() - logits.min())
        #
        # # print('Prediction:', prediction.shape, prediction)
        #
        # prediction = self.manage_preds(prediction)
        # # print('Prediction:', prediction)
        # # print('Logits: ', logits.shape)
        #
        # return prediction, logits


    
