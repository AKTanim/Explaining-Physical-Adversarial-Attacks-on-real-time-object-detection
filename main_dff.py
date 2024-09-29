import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import numpy as np
import requests
import cv2
import json
import torch
import torch.nn as nn
from utils_dff.deep_feature_factorization import DeepFeatureFactorization
from utils_dff.image import show_cam_on_image, preprocess_image, deprocess_image, show_factorization_on_image
from utils_dff.grad_cam import GradCAM
from torchvision.models import resnet50
from utils_dff.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from models.experimental import attempt_load

def load_image(path):
    """A function that loads an image,
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(path))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

def create_labels(concept_scores, top_k=2):
    """ Create a list with the image-net or coco category names of the top scoring categories.
    Resnet is pretrained on Imagenet. To use coco, use a custom classifier.
    """
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    # print(f"Labels: {labels}")
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    print(f"concept categories{concept_categories.shape}: {concept_categories}")
    concept_labels_topk = []

    # coco_labels_url = "https://raw.githubusercontent.com/Nebula4869/YOLOv5-LibTorch/master/coco.names"
    # labels = requests.get(coco_labels_url).text.splitlines()
    # # print(f"Labels: {labels}")
    # concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    # print(f"concept categories{concept_categories.shape}: {concept_categories}")
    # concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = []
        for category in categories:
            # print(f"category: {category}")
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


def visualize_image(model, img_path, n_components=2, top_k=2):
    # img, rgb_img_float, input_tensor = load_image(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    rgb_img_float = np.float32(img) / 255
    print(f"rgb img size{type(rgb_img_float)}: {rgb_img_float.shape}")
    input_tensor = model.preprocessing(img[..., ::-1])
    # classifier = resnet50(pretrained=True)
    # classifier.eval()
    # classifier = classifier.fc # Todo?
    print("Model:\n", model)
    target_layer = model.model.model._modules['9'].cv2
    classifier = YOLOv5Classifier(target_layer, num_classes=1000) # 80 for coco, 1000 for imagenet
    classifier.eval()
    classifier = classifier.fc3
    # Decomposed activations (into components) may not inherently correspond to any specific classes or objects.
    # The classifier helps in interpreting these components by assigning them to meaningful categories
    print(f"Target layer: {model.model.model._modules['23'].cv3}")
    print(f"classifier: {classifier}")
    dff = DeepFeatureFactorization(model=model, target_layer=model.model.model._modules['23'].cv3._modules['act'],
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)
    print(f"Concept_outputs: {concept_outputs.shape}")

    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.3,
                                                concept_labels=concept_label_strings)

    result = np.hstack((img, visualization))

    # Just for the jupyter notebook, so the large images won't weight a lot:
    # if result.shape[0] > 500:
    #     result = cv2.resize(result, (result.shape[1] // 4, result.shape[0] // 4))

    return result

class YOLOv5Classifier(nn.Module):
    def __init__(self, target_layer, num_classes=80):
        super(YOLOv5Classifier, self).__init__()
        self.target_layer = target_layer
        self.fc1 = nn.Linear(512 * 1 * 1, 1024)  # Assuming input features match output of target layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.target_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# classifier = resnet50(pretrained=True)
# classifier.eval()
model_path = 'yolov5s.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
names = None
model = YOLOV5TorchObjectDetector(model_path, device, img_size=(640, 640),
                                      names=None if names is None else names.strip().split(","))
res_img = visualize_image(model, "images/inria(5).png")
# converting BGR to RGB
# res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs/dff_img.png', res_img)








# import warnings
#
# warnings.filterwarnings('ignore')
# from PIL import Image
# import numpy as np
# import requests
# import cv2
# import torch
# from sklearn.decomposition import NMF
# from utils_dff.deep_feature_factorization import DeepFeatureFactorization
# from utils_dff.image import preprocess_image, show_factorization_on_image
# from utils_dff.grad_cam import GradCAM
# # from yolov5 import YOLOv5  # Assuming YOLOv5 class is imported from the correct module
#
#
# def load_image(path):
#     """A function that loads an image,
#     and returns a numpy image and a preprocessed
#     torch tensor ready to pass to the model """
#     img = np.array(Image.open(path))
#     rgb_img_float = np.float32(img) / 255
#     input_tensor = preprocess_image(rgb_img_float,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     return img, rgb_img_float, input_tensor
#
#
# def create_labels(concept_scores, top_k=2):
#     """ Create a list with the COCO category names of the top scoring categories """
#     coco_labels_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names"
#     labels = requests.get(coco_labels_url).text.splitlines()
#     concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
#     concept_labels_topk = []
#     for concept_index in range(concept_categories.shape[0]):
#         categories = concept_categories[concept_index, :]
#         concept_labels = []
#         for category in categories:
#             score = concept_scores[concept_index, category]
#             label = f"{labels[category].split(',')[0]}:{score:.2f}"
#             concept_labels.append(label)
#         concept_labels_topk.append("\n".join(concept_labels))
#     return concept_labels_topk
#
#
# def dff(activations: np.ndarray, n_components: int = 5):
#     """ Compute Deep Feature Factorization on a 2D Activations tensor."""
#     batch_size, channels, h, w = activations.shape
#     reshaped_activations = activations.transpose((1, 0, 2, 3))
#     reshaped_activations[np.isnan(reshaped_activations)] = 0
#     reshaped_activations = reshaped_activations.reshape(reshaped_activations.shape[0], -1)
#     offset = reshaped_activations.min(axis=-1)
#     reshaped_activations = reshaped_activations - offset[:, None]
#
#     model = NMF(n_components=n_components, init='random', random_state=0)
#     W = model.fit_transform(reshaped_activations)
#     H = model.components_
#     concepts = W + offset[:, None]
#     explanations = H.reshape(n_components, batch_size, h, w)
#     explanations = explanations.transpose((1, 0, 2, 3))
#     return concepts, explanations
#
#
# class YOLOv5WithActivations():
#     def __init__(self, model_path):
#         super().__init__(model_path)
#         self.activations = None
#
#     def forward(self, x, augment=False, profile=False):
#         self.activations = None
#
#         def hook(module, input, output):
#             self.activations = output
#
#         layer = self.model.model[-2]  # Example: second last layer
#         handle = layer.register_forward_hook(hook)
#         output = super().forward(x, augment, profile)
#         handle.remove()
#         return output
#
#
# class YOLOv5Classifier(torch.nn.Module):
#     def __init__(self, num_classes=80):  # COCO has 80 classes
#         super().__init__()
#         self.classifier = torch.nn.Linear(2048, num_classes)  # Assuming 2048 input features
#
#     def forward(self, x):
#         return self.classifier(x)
#
#
# def visualize_image(model, img_path, n_components=5, top_k=2):
#     img, rgb_img_float, input_tensor = load_image(img_path)
#     classifier = YOLOv5Classifier(num_classes=80)
#     dff = DeepFeatureFactorization(model=model, target_layer=model.model[-2], computation_on_concepts=classifier)
#     concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)
#
#     concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
#     concept_label_strings = create_labels(concept_outputs, top_k=top_k)
#     visualization = show_factorization_on_image(rgb_img_float,
#                                                 batch_explanations[0],
#                                                 image_weight=0.3,
#                                                 concept_labels=concept_label_strings)
#
#     result = np.hstack((img, visualization))
#     return result
#
#
# model_path = 'yolov5s.pt'  # Path to your YOLOv5 model
# model = YOLOv5WithActivations(model_path)
# model.eval()
# print("Loaded YOLOv5 model")
#
# res_img = visualize_image(model, "images/inria(5).png")
# # converting BGR to RGB
# res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
# cv2.imwrite('outputs/dff_img.png', res_img)
