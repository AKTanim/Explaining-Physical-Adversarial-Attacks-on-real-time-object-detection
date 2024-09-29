import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV2GradCAM
from models.yolo_v2_od import YOLOv2OD
import cv2
from deep_utils import Box, split_extension
from utils.datasets import letterbox
import torch

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="coco", help='Path to the model: coco or pascal')
parser.add_argument('--img-path', type=str, default='images/inria/adv/atk_images/yolo_v2_final', help='input image path')
parser.add_argument('--output-dir', type=str, default='outputs/cam_test/yolo_v2_final', help='output dir')
parser.add_argument('--img-size', type=int, default=544, help="input image size: 544 or 416")
# parser.add_argument('--target-layer', type=str, default='model_23_cv3_act',
#                     help='The layer hierarchical address to which gradcam will applied,'
#                          ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')

args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    # fills the area outside the bounding box (bbox) with zeros or some specified values in the heatmap.
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def put_text_box(bbox, cls_name, conf, res_img):
    x1, y1, x2, y2 = bbox
    lbl_name = '{0}: {1}'.format(cls_name, conf)
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, lbl_name, (x1, y1))
    return res_img


def concat_images(images, img_path):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='_step')
    for i, img in enumerate(images):
        if i != 0:
            path = f'{args.output_dir}/{img_name}_{i}.png'
            cv2.imwrite(path, img)
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img

def yolo_resize(img, new_shape=(544, 544), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):

    return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

def preprocessing(img):
    if len(img.shape) != 4:
        img = np.expand_dims(img, axis=0)
    im0 = img.astype(np.uint8)
    img = np.array([yolo_resize(im)[0] for im in im0])
    img = img.transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(args.device)
    img = img / 255.0
    return img


def main(img_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    img = cv2.imread(img_path)
    print('[INFO] Loading the model')
    model = YOLOv2OD(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))

    torch_img = preprocessing(img[..., ::-1])
    if args.method == 'gradcam':
        saliency_method = YOLOV2GradCAM(model=model, img_path=img_path, img_size=input_size)
        # saliency_method = YOLOV8GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    tic = time.time()
    masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img)
    # print(f"Maks type: {type(masks)}")
    # masks, boxes, class_names = saliency_method(torch_img)
    # torch_img = saliency_method.preprocessing(img[..., ::-1])
    print('torch_img_size:', torch_img.shape)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]
    print("Img size: ", result.shape)
    for i, mask in enumerate(masks):
        # print(f"Mask({i}): {mask.shape}")
        res_img = result.copy()
        bbox, cls_name, conf = boxes[0][i], class_names[0][i], confs[0][i] # [y1, x1, y2, x2]
        # print(f"Bbox({i})_{cls_name}: {bbox}")
        res_img, heat_map = get_res_img(bbox, mask, res_img)
        res_img = put_text_box(bbox, cls_name, conf, res_img)
        images.append(res_img)
    final_image = concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='_gradcam')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)


def folder_main(folder_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    print('[INFO] Loading the model')
    model = YOLOv2OD(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    for item in os.listdir(folder_path):
        img_path = os.path.join(folder_path, item)
        img = cv2.imread(img_path) # reads in BGR format and [h, w, c]
        torch_img = preprocessing(img[..., ::-1])
        # ... takes all preceding elements. ::-1 reverses the last dimension, hence BGR->RGB
        if args.method == 'gradcam':
            saliency_method = YOLOV2GradCAM(model=model, img_path=img_path, img_size=input_size)
        tic = time.time()
        masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img) # returns sal map, logits, preds
        print("total time:", round(time.time() - tic, 4))
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # squeeze(0): Removes the batch dimension from the tensor.
        # mul(255).add_(0.5).clamp_(0, 255): Scales the image pixel values from the range [0, 1] to [0, 255] and clamps them to this range.
        # permute(1, 2, 0): Rearranges the dimensions from (C, H, W) to (H, W, C) for image display.
        # detach().cpu().numpy(): Detaches the tensor from the computation graph and converts it to a NumPy array.
        # result[..., ::-1]: Converts the image from RGB to BGR format for OpenCV.
        images = [result]
        print("Img size: ", result.shape)
        for i, mask in enumerate(masks): # one mask per detected bbox
            # print(f"Mask({i}): {mask.shape}")
            res_img = result.copy()
            bbox, cls_name, conf = boxes[0][i], class_names[0][i], confs[0][i] # [0] represents 1st image since there's a batch dimension
            res_img, heat_map = get_res_img(bbox, mask, res_img)
            res_img = put_text_box(bbox, cls_name, conf, res_img)
            images.append(res_img) # one image per detected bbox
        final_image = concat_images(images, img_path)
        img_name = split_extension(os.path.split(img_path)[-1], suffix='_gradcam')
        output_path = f'{args.output_dir}/{img_name}'
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'[INFO] Saving the final image at {output_path}')
        cv2.imwrite(output_path, final_image)


if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        folder_main(args.img_path)
    else:
        main(args.img_path)
