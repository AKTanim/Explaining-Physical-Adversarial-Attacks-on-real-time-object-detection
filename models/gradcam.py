import time
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from utils.datasets import letterbox
from inference.utils_yolov2.util_grad import *

def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]
    # print(target_layer)

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
        # print(f"Target layer{[h]}: {target_layer}")
    print('Target:', target_layer)
    return target_layer

def find_rtdetr_layer(model):
    """Find rtdetr layers to calculate GradCAM and GradCAM++

    Args:
        model: RtDETR model.
    Return:
        target_layer: found layer
    """
    target_layers = []
    for l in range(5):
        target_layer = model.decoder.dec_bbox_head._modules[str(l)].act
        target_layers.append(target_layer)
        # print(f"Target layer{[h]}: {target_layer}")
    print(f"Target layers{type(target_layers)}: {target_layers}")
    return target_layers


class YOLOV2GradCAM:

    def __init__(self, model, img_path, img_size=(544, 544)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.img_path = img_path
        self.img_size = img_size

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            # print('grad_in: ', torch.nonzero(grad_input[0]))
            # print('module: ', module)
            # print('grad_out: ', torch.nonzero(grad_output[0]))
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            # print('F_grad_in: ', torch.nonzero(input[0]))
            # print('F_module: ', module)
            # print('F_grad_out: ', torch.nonzero(output[0]))
            return None

        print("Target layer:\n", self.model.model.module_list._modules['30']._modules)
        target_layer = self.model.model.module_list._modules['29']._modules['leaky_29']
        print('Target:', target_layer)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        print('target_f_h: ', target_layer.register_forward_hook(forward_hook))
        print('target_b_h: ', target_layer.register_backward_hook(backward_hook))

        self.device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        # img = cv2.imread(self.img_path)
        # self.model(self.preprocessing(img[..., ::-1]))
        self.model(torch.ones(1, 3, *img_size, device=self.device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        # input_img = self.preprocessing(input_img[..., ::-1])
        b, c, h, w = input_img.size()
        print("Image size:", input_img.size())
        tic = time.time()
        # from YOLO v2
        # preds [no of det * 8] -> [[batch, x1, y1, x2, y2, conf, objectness, class_id],[...],...]
        # logits [batch, 1445, 85] -> raw scores (1 objectness, 4 bbox, 80 confs)
        score, anchors, logits = self.model(input_img)
        prediction = predict_transform(score, 544, 32, anchors,
                                       80, 0.5, CUDA=torch.cuda.is_available())
        # print('pred_m_1: ', len(prediction), len(prediction[0]), len(prediction[0][0]))
        prediction, _ = write_results(prediction, 80, nms=True, nms_conf=0.4)
        prediction = prediction.view(-1, 8)
        preds = self.manage_preds(prediction)
        # Needed for gradcam
        # preds [4, batch, no of det] -> [[bbox] [class id] [cls names] [confidence]]
        # logits [batch, no of det, 80] -> raw unnormalized score before softmax. logits > activation > probability
        # preds, logits = self.manage_score(preds, logits, input_img)
        # print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        print(f"Preds: {preds}")
        # print(f"Logits: {len(logits)}*{len(logits[0])}*{len(logits[0][0])} {logits}")
        print(f"Detected {len(preds[1][0])} objects:")
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if cls_name == 'person':
                if class_idx:
                    score = logit[cls]
                    print(f"Score(logit[{cls}-{cls_name}]) {type(score)} grad-{score.requires_grad}: {score}")
                else:
                    score = logit.max()
                # self.model.zero_grad() # clears the previous gradients in the model
                tic = time.time()
                score.backward(retain_graph=True)
                # computes the gradients of the score with respect to the model's input.
                # retain_graph=True flag is used to retain the computation graph for further backward passes.
                print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
                # print(f"Self_grad {list(self.gradients.values())[0].size()}")
                # print('self_grad', torch.nonzero(self.gradients['value']))
                gradients = self.gradients['value']
                activations = self.activations['value']
                # print(f"Gradients {gradients.shape}: {torch.nonzero(gradients)}")
                # print(f"Activations {activations.shape}: {activations}")

                b, k, u, v = gradients.size() # batch, channel, height, width
                alpha = gradients.view(b, k, -1).mean(2) # flattens u,v->u*v and takes mean per channel
                weights = alpha.view(b, k, 1, 1) # reshaping

                saliency_map = (weights * activations).sum(1, keepdim=True)
                # elementwise multiplication. Then sum in the k dim which collapses the 3 channels into one.
                # reulting in a single-channel saliency map per image
                saliency_map = F.relu(saliency_map) # retains non-negative values. only positive contribution.
                # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # deprecated
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=True)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                # print("sal_min_max: ", saliency_map_min, "\t", saliency_map_max)
                # epsilon = 1e-7
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data # normalized
                saliency_maps.append(saliency_map)
        print(f"Saliency maps: {np.asarray(saliency_maps).shape}")
        # print(saliency_maps)
        return saliency_maps, logits, preds

    @staticmethod
    def yolo_resize(img, new_shape=(544, 544), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):

        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        img = np.array([self.yolo_resize(im, new_shape=self.img_size)[0] for im in im0])
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        return img


    def manage_preds(self, preds):
        classes = load_classes('inference/utils_yolov2/data/coco.names')
        bbox = []
        cls_ids = []
        cls_names = []
        confs =[]
        for pred in preds:
            x1, y1, x2, y2 = pred[1].int().item(), pred[2].int().item(),pred[3].int().item(), pred[4].int().item()
            box = [abs(y1), abs(x1), abs(y2), abs(x2)]
            cls_id = pred[-1].int().item()
            cls_name = classes[cls_id]
            conf = float('{0:.2f}'.format(pred[-3].item()))

            bbox.append(box)
            cls_ids.append(cls_id)
            cls_names.append(cls_name)
            confs.append(conf)

        preds_out = [[bbox], [cls_ids], [cls_names], [confs]]
        return preds_out

    def __call__(self, input_img):
        return self.forward(input_img)

class YOLOV5GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            # print('grad_in: ', torch.nonzero(grad_input[0]))
            # print('module: ', module)
            # print('grad_out: ', torch.nonzero(grad_output[0]))
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            # print('F_grad_in: ', torch.nonzero(input[0]))
            # print('F_module: ', module)
            # print('F_grad_out: ', torch.nonzero(output[0]))
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape)#shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        # print("Image size:", input_img.size())
        tic = time.time()
        preds, logits = self.model(input_img)
        # preds -> [[bbox] [class id] [cls names] [confidence]]
        # logits -> raw unnormalized score before softmax. logits > activation > probability
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        print(f"Preds {len(preds)}*{len(preds[0])}*{len(preds[0][0])}: {preds}")
        print(f"Logits {len(logits)}*{len(logits[0])}*{len(logits[0][0])}: {logits}")
        print(f"Detected {len(preds[1][0])} objects:")
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if cls_name == 'person':
                if class_idx:
                    score = logit[cls]
                    print(f"Score(logit[{cls}-{cls_name}]) {type(score)} grad-{score.requires_grad}: {score}")
                else:
                    score = logit.max()
                self.model.zero_grad() # clears the previous gradients in the model
                tic = time.time()
                score.backward(retain_graph=True)
                # computes the gradients of the score with respect to the model's input.
                # retain_graph=True flag is used to retain the computation graph for further backward passes.
                print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
                # print(f"Self_grad {list(self.gradients.values())[0].size()}")
                gradients = self.gradients['value']
                activations = self.activations['value']
                print(f"Gradients {gradients.shape}: {torch.nonzero(gradients)}")
                # print(f"Activations {activations.shape}: {activations}")

                b, k, u, v = gradients.size() # batch, channel, height, width
                alpha = gradients.view(b, k, -1).mean(2) # flattens u,v->u*v and takes mean per channel
                weights = alpha.view(b, k, 1, 1) # reshaping

                saliency_map = (weights * activations).sum(1, keepdim=True)
                # elementwise multiplication. Then sum in the k dim which collapses the 3 channels into one.
                # reulting in a single-channel saliency map per image
                saliency_map = F.relu(saliency_map) # retains non-negative values. only positive contribution.
                # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # deprecated
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data # normalized
                saliency_maps.append(saliency_map)
        print(f"Saliency maps: {np.asarray(saliency_maps).shape}")
        # print(saliency_maps)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        return self.forward(input_img)


class rtdetrCAM:

    def __init__(self, model, module, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.gradient_list = []
        self.activation_list = []
        self.module = module

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            self.gradient_list.append(self.gradients)
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            self.activation_list.append(self.activations)
            return None

        # target_layer = find_rtdetr_layer(self.model) # todo ?
        # target_layer = self.model.decoder.dec_bbox_head._modules['0'].act
        # target_layer = self.model.decoder.enc_bbox_head._modules['act']
        # target_layer = self.model.encoder._modules['pan_blocks']._modules['1'].conv2.act
        # target_layer = self.model.encoder._modules['pan_blocks']._modules['1'].bottlenecks._modules['2'].act
        target_layer = model.decoder.dec_bbox_head._modules[str(self.module)].act
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def manage_score(self, outputs, orig_image):
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

        logits_list = []
        boxes = []
        cls_names = []
        cls_ids = []
        confidences = []

        for i in range(len(outputs['pred_boxes'][0])):  # loop per object box (total 300)
            logits = outputs['pred_logits'][0][i]  # extraxt 80 values of each object
            # print('Logit: ', logits)
            soft_logits = torch.softmax(logits, dim=-1)  # probability
            max_index = torch.argmax(soft_logits).cpu()  # max confidence out of the 80 values
            class_name = mscoco_category2name[mscoco_label2category[int(max_index)]]

            if soft_logits[max_index] > 0.50:  # filters out low probability detections out of 300 detections
                logits_list.append(logits)

                cx, cy, w, h = outputs['pred_boxes'][0][i].cpu().detach().numpy().tolist()
                cx = cx * orig_image.shape[3]
                cy = cy * orig_image.shape[2]
                w = w * orig_image.shape[3]
                h = h * orig_image.shape[2]
                x1 = int(cx - (w // 2))
                y1 = int(cy - (h // 2))
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                box = [y1, x1, y2, x2] # [y1, x1, y2, x2]

                cls_name = class_name
                cls_id = mscoco_category2label[mscoco_label2category[int(max_index)]]
                confidence = soft_logits[max_index].detach().numpy().tolist()
                confidence = round(confidence, 2)  # or "{:.2f}".format(confidence)

                boxes.append(box)
                cls_names.append(cls_name)
                cls_ids.append(cls_id)
                confidences.append(confidence)

        preds = [[boxes], [cls_ids], [cls_names], [confidences]]
        logits_tensor = torch.stack(logits_list)
        return preds, logits_tensor

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()

        outputs = self.model(input_img)
        print('before manage score:', outputs['pred_logits'].shape, outputs['pred_logits'])
        preds, logits = self.manage_score(outputs, input_img)
        # preds -> [[bbox] [class id] [cls names] [confidence]]
        # logits -> raw unnormalized score before softmax. logits > activation > probability
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        print(f"Preds: {preds}")
        print(f"Logits {logits.shape}: {logits[0]}")
        print(f"Detected {len(preds[1][0])} objects")
        for logit, cls, cls_name in zip(logits, preds[1][0], preds[2][0]):
            if cls_name == 'person':
                if class_idx:
                    score = logit[cls]
                    # print(f"Score(logit[{cls}-{cls_name}]): {score}")
                else:
                    score = logit.max()
                self.model.zero_grad() # clears the previous gradients in the model
                tic = time.time()
                score.backward(retain_graph=True)
                # computes the gradients of the score with respect to the model's input.
                # retain_graph=True flag is used to retain the computation graph for further backward passes.
                print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
                # print(f"Self_grad {list(self.gradients.values())[0].size()}")
                gradients = self.gradients['value']
                activations = self.activations['value']

                gradients = gradients.view(gradients.shape[0], gradients.shape[1], 16, 16)
                activations = activations.view(activations.shape[0], activations.shape[1], 16, 16)
                print(f"Gradients {gradients.shape}")
                print(f"Activations {activations.shape}")

                b, k, u, v = gradients.size() # batch, channel, height, width
                alpha = gradients.view(b, k, -1).mean(2) # flattens u,v->u*v and takes mean per channel
                weights = alpha.view(b, k, 1, 1) # reshaping

                saliency_map = (weights * activations).sum(1, keepdim=True)
                # elementwise multiplication. Then sum in the k dim which collapses the channels into one.
                # reulting in a single-channel saliency map per image
                saliency_map = F.relu(saliency_map) # retains non-negative values. only positive contribution.
                # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # deprecated
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data # normalized
                saliency_maps.append(saliency_map)
        print(f"Saliency maps: {np.asarray(saliency_maps).shape}")
        # print(saliency_maps)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        return self.forward(input_img)


class YOLOV8GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()

        # preds, logits = self.model(input_img)
        # preds -> [[bbox] [class id] [cls names] [confidence]]
        # logits -> raw unnormalized score before softmax. logits > activation > probability
        results = self.model(input_img)
        print(input_img)
        bbox = results[0].boxes.xyxy
        conf = results[0].boxes.conf
        cls_names = []

        for id in results[0].boxes.cls.int().tolist():
            name = results[0].names[id]
            cls_names.append(name)

        print(f"Boxes: {bbox}\nClass names{type(cls_names)}: {cls_names}")

        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        # print(f"Preds: {preds}")
        # print(f"Detected {len(preds[1][0])} objects:")

        for score, cls_name in zip(conf, cls_names):
            # if class_idx:
            #     score = logit[cls]
            #     print(f"Score(logit[{cls}-{cls_name}]): {score}")
            # else:
            #     score = logit.max()
            self.model.zero_grad() # clears the previous gradients in the model
            tic = time.time()
            score.backward(retain_graph=True)
            # computes the gradients of the score with respect to the model's input.
            # retain_graph=True flag is used to retain the computation graph for further backward passes.
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value']
            activations = self.activations['value']

            b, k, u, v = gradients.size() # batch, channel, height, width
            alpha = gradients.view(b, k, -1).mean(2) # flattens u,v->u*v and takes mean per channel
            weights = alpha.view(b, k, 1, 1) # reshaping

            saliency_map = (weights * activations).sum(1, keepdim=True)
            # elementwise multiplication. Then sum in the k dim which collapses the 3 channels into one.
            # reulting in a single-channel saliency map per image
            saliency_map = F.relu(saliency_map) # retains non-negative values. only positive contribution.
            # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # deprecated
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=True)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data # normalized
            saliency_maps.append(saliency_map)
        return saliency_maps, bbox, cls_names

    def __call__(self, input_img):
        return self.forward(input_img)
