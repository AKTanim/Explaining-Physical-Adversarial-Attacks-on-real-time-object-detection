import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pprint

#from darknet import Darknet

from utils_adv.median_pool import MedianPool2d  # see median_pool.py

class yolov5_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov5_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, logits, loss_type):
        results = []

        for logit in logits:
            conf_normal = logit

            obj_scores = torch.sum(conf_normal[:, :], dim=1)
            confs_if_object_normal = conf_normal[:, self.cls_id]
            person_cls_score = confs_if_object_normal / obj_scores

            loss_target = person_cls_score

            if loss_type == 'max_approach':
                if loss_target.numel() == 0:  # Check if the tensor is empty
                    print("Warning: loss_target is empty. Skipping max operation.")
                    max_conf = torch.tensor(0, device=logit.device, dtype=logit.dtype)
                else:
                    max_conf, max_conf_idx = torch.max(loss_target, dim=0)  # take the maximum value
                results.append(max_conf)
            elif loss_type == 'threshold_approach':
                threshold = 0.35
                penalized_tensor = torch.max(confs_if_object_normal - threshold, torch.zeros_like(confs_if_object_normal)) ** 2
                thresholded_conf = torch.sum(penalized_tensor, dim=0)
                results.append(thresholded_conf)

        # Stack the results into a single tensor to return
        return torch.stack(results)



# class yolov5_feature_output_manage(nn.Module):
#     """MaxProbExtractor: extracts max class probability for class from YOLO output.
#
#     Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
#
#     """
#
#     def __init__(self, cls_id, num_cls, config):
#         super(yolov5_feature_output_manage, self).__init__()
#         self.cls_id = cls_id
#         self.num_cls = num_cls
#         self.config = config
#
#     def forward(self, preds, loss_type):
#         # get values necessary for transformation
#         print(f"***YOLO {type(preds)}: {preds.shape}")
#
#         output_objectness_not_norm = preds[:, :, 4]
#         output_objectness_norm = torch.sigmoid(preds[:, :, 4])  # [batch, 1, 845]  # iou_truth*P(obj)
#         # take the fifth value, i.e. object confidence score. There is one value for each box, in total 5 boxes
#
#         output = preds[:, :, 5: 5+self.num_cls]  # [batch, 80, 845]  # 845 = 5 * h * w
#         # NB 80 means conditional class probabilities, one for each class related to a single box (there are 5 box for each grid cell)
#
#         # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
#         not_normal_confs = output
#         normal_confs = torch.nn.Softmax(dim=2)(output)
#         # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1
#
#         # we only care for conditional probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
#         confs_for_class_not_normal = not_normal_confs[:, :, self.cls_id]
#         confs_for_class_normal = normal_confs[:, :, self.cls_id] # take class number 0, so just one kind of cond. prob out of 80. This is for 1 box, there are 5 boxes
#
#         confs_if_object_not_normal = self.config.loss_target(output_objectness_not_norm, confs_for_class_not_normal)
#         confs_if_object_normal = self.config.loss_target(output_objectness_norm, confs_for_class_normal)  # loss_target in patch_config
#
#         if loss_type == 'max_approach':
#             max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
#             return max_conf
#
#         elif loss_type == 'threshold_approach':
#             threshold = 0.3
#             batch_stack = torch.unbind(confs_if_object_normal, dim=0)
#             print('yolo batch stack: \n')
#             print(batch_stack)
#             penalized_tensor_batch = []
#             for img_tensor in batch_stack:
#                 size = img_tensor.size()
#                 zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
#                 penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
#                 penalized_tensor_batch.append(penalized_tensor)
#
#             penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
#             thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
#             return thresholded_conf


class yolov2_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov2_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput, loss_type):
        # get values necessary for transformation

        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)  # add one dimension of size 1
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (
                    5 + self.num_cls) * 5)  # the last 5 is the anchor boxes/ priors after k-means clustering
        # the first 5 is the number of parameters of each box: x, y, w, h, and, objectness score
        # self.num_cls indicates class probabilities, i.e. 20 values for VOC and 80 for COCO
        # in total, there are 125 parameters per grid cell when VOC, 425 when COCO
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # print('dim0:' + str(YOLOoutput.size(0)))
        # print('dim1:' + str(YOLOoutput.size(1)))
        # print('h:' + str(h))
        # print('w:' + str(h))

        # transform the output tensor from [batch, 5*85, 13, 13] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1,
                                  2).contiguous()  # [batch, 85, 5, 169] swap 5 and 85, in position 1 and 2 respectively
        output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 845]
        # todo first 5 numbers that make '85' are box xc, yc, w, h and objectness. Last 80 are class prob.

        # print(output[:, 4, :])
        # print(output[:, 5, :])
        # print(output[:, 6, :])
        # print(output[:, 34, :])

        output_objectness_not_norm = output[:, 4, :]
        output_objectness_norm = torch.sigmoid(output[:, 4, :])  # [batch, 1, 845]  # iou_truth*P(obj)
        # take the fifth value, i.e. object confidence score. There is one value for each box, in total 5 boxes

        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 845]  # 845 = 5 * h * w
        # NB 80 means conditional class probabilities, one for each class related to a single box (there are 5 box for each grid cell)

        # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
        not_normal_confs = output
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1

        # we only care for conditional probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
        confs_for_class_not_normal = not_normal_confs[:, self.cls_id, :]
        confs_for_class_normal = normal_confs[:, self.cls_id,
                                 :]  # take class number 0, so just one kind of cond. prob out of 80. This is for 1 box, there are 5 boxes

        confs_if_object_not_normal = self.config.loss_target(output_objectness_not_norm, confs_for_class_not_normal)
        confs_if_object_normal = self.config.loss_target(output_objectness_norm,
                                                         confs_for_class_normal)  # loss_target in patch_config

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal,
                                               dim=1)  # take the maximum value among your 5 priors
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('yolo batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf
            

class ssd_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(ssd_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, ssd_output, loss_type):
        # get values necessary for transformation
        # conf_normal, conf_not_normal, loc = ssd_output
        conf_normal = ssd_output
        # print("Check scores: ---------------------------")
        # print(f"Confidence normal: {conf_normal.shape}\n{conf_normal}")
        # print(f"Location {loc.shape}: {loc}")

        #obj_scores = torch.sum(conf_normal[:,:,1:], dim=2)
        obj_scores = torch.sum(conf_normal[:,:,1:], dim=2)
        # print(f"Object score {obj_scores.shape}: {obj_scores}")
                                                                                         
        confs_if_object_normal = conf_normal[:, :, self.cls_id] # softmaxed #obj*cls   
        # print(f"confs_if_object_normal {confs_if_object_normal.shape}: {confs_if_object_normal}")
                                                                                         
        person_cls_score = confs_if_object_normal / obj_scores  
        # print(f"person_cls_score {person_cls_score.shape}: {person_cls_score}")
                                                                                         
        #loss_target = confs_if_object_normal                   
        loss_target = person_cls_score                                             

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(loss_target, dim=1) # take the maximum value among your 5 priors
            # print(f"Max_conf {max_conf.shape}: {max_conf}")
            return max_conf
        elif loss_type == 'threshold_approach':
            threshold = 0.35
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            #print('ssd batch stack: \n')
            #print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor)**2
                #penalized_tensor = penalized_tensor**2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)
        # Gradient will not be calculated during backpropagation, but nn.parameter makes it a learnable parameter

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference

        color_dist = (adv_patch - self.printability_array+0.000001) # Adding ).000001 to avoid a value of zero
        # 30 3D arrays
        #print(color_dist.size())
        color_dist = color_dist ** 2  # squared difference
        color_dist = torch.sum(color_dist, 1)+0.000001 # color channels will be merged pixel-wise for all 30 images > 30*1*300*300
        #print(color_dist.size())
        color_dist = torch.sqrt(color_dist)

        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # torch.min(color_dist, 0) returns a tuple of 2, 1st being the values and 2nd being their indices
        # After this operation, the color_dist_prod will ne 1 image achieved from the pixel-wise min value of the 30 images > 1*1*300*300
        #print(type(color_dist_prod))
        #print('size ' + str(color_dist_prod.size()))

        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)  # divide by the total number of elements in the input tensor

    def get_printability_array(self, printability_file, side):
        #  side = patch_size in adv_examples.py
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        # see notes for a better graphical representation
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))

            printability_array.append(printability_imgs) # dim: 30*3*(300*300)

        printability_array = np.asarray(printability_array)  # convert input lists, tuples etc. to array
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)  # Creates a Tensor from a numpy array.
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # compute total variation of the adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)

        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

        
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7*7
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False):

        use_cuda = 0

        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))  # pre-processing on the image with 1 more dimension: 1 x 3 x 300 x 300, median_pool.py expects a 4D input

        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2  # img_size = 416, adv_patch size = patch_size in adv_examples.py, = 300
        # print('pad =' + str(pad)) # pad = 0.5*(416 - 300) = 58

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        # print('adv_patch in load_data.py, PatchTransforme, size =' + str(adv_patch.size()))
        print(f"Lab_batch size: {lab_batch.size()}")
        # Lab_batch size: torch.size([8,14,5])
        # adv_patch in load_data.py, PatchTransforme, size =torch.Size([1, 1, 3, 300, 300]), tot 5 dimensions

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        #print('adv_batch in load_data.py, PatchTransforme, size =' + str(adv_batch.size()))
        # adv_batch in load_data.py, PatchTransforme, size =torch.Size([8, 14, 3, 300, 300])

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        #print('batch_size in load_data.py, PatchTransforme, size =' + str(batch_size))
        # batch_size in load_data.py, PatchTransforme, size =torch.Size([8, 14])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor

        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            # Fills self tensor (here 8 x 14) with 'random' numbers sampled from the continuous uniform distribution: 1/(max_contrast - min_contrast)

        # print('contrast1 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast1 in load_data.py, PatchTransforme, size =torch.Size([8, 14])

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print('contrast2 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast2 in load_data.py, PatchTransforme, size =torch.Size([8, 14, 1, 1, 1])

        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # -1 in pytorch.expand indicates that the size along that dimension will remain unchanged i.e. 8 and 14
        # print('contrast3 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast3 in load_data.py, PatchTransforme, size =torch.Size([8, 14, 3, 300, 300])

        # lines 206-221 could be replaced by:
        # contrast = torch.FloatTensor(adv_batch).uniform_(self.min_contrast, self.max_contrast)
        # print('contrast4 in load_data.py, PatchTransforme, size =' + str(contrast.size()))

        ### Ultimately this creates 8*14 pieces of matrices(size: 3*300*300 like an image) with random contrasts per pixel

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        # lines 227-239 could be replaced by:
        # brightness = torch.FloatTensor(adv_batch).uniform_(self.min_brightness, self.max_brightness)
        # print('brightness in load_data.py, PatchTransforme, size =' + str(brightness.size()))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # dim: 8 x 14 x 3 x 300 x 300
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise # vectorized operation. All variables with same dim [8,14,3,300,300].

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # Clips values of all elements in the range [0.000001,0.99999] (real numbers since FLoatTensor)
        # dim: 8 x 14 x 3 x 300 x 300

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # extracts dimension 2 from lab_batch (from [8,14,5] takes 5)
                                                    # then takes the 1st value starting from 0 taking 1 value (which is class id(0 for person) from yolo_label)
                                                    # Consider just the first 'column' of lab_batch, where we can
                                                    # discriminate between detected person (or 'yes person'= 0) and 'no person'= 1)
                                                    # in this way, sensible data about x, y, w and h of the rectangles are not used for building the mask

        # torch.narrow(arr, dim, start, length) -> remove 'arr' through dimension 'dim' and keep 'arr' from 'start' upto length 'length'
        # NB torch.narrow returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length.
        # The returned tensor and input tensor share the same underlying storage.
        print(f'cls_ids size from load.py:{cls_ids.size()}')
        # cls_ids size from load.py:  torch.Size([8, 14, 1])

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # 8 x 14 x 3 x 300 x 300

        if use_cuda:
            # msk_batch = cls_mask - torch.cuda.FloatTensor(cls_mask.size()).fill_(0)
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask # cls_mask has all zero values for yolo
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # take a matrix of 1s, subtract that of the labels so that
                                                                                # we can have 0s where there is no person detected,
                                                                                # obtained by doing 1-1=0
                                                                                # This works because we added 1s when creating lab_batch with max 14 persons

        # NB! Now the mask has 1s 'above', where the labels data are sensible since they represent detected persons, and 0s where there are no detections
        # In this way, multiplying the adv_batch to this mask, built from the lab_batch tensor, allows to target only detected persons and nothing else,
        # i.e. pad with zeros the rest
        #print("Cls_mask: ",cls_mask)
#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)  # dim 8 x 14 x 3 x 416 x 416
        msk_batch = mypad(msk_batch)  # dim 8 x 14 x 3 x 416 x 416

        # NB you see only zeros when you print it because they are all surrounding the patch to pad it to image dimensions (3 x 416 x 416)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 8*14 = 112
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle) # creates 112 angle values

        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)  # 300

        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 8 x 14 x 5

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2)) # Patch size
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # np.prod(batch_size) = 8*14 = 112
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y

        if(rand_loc): # if True, puts patches randomly instead of putting them on the person
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        target_y = target_y - 0.05

        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size() # 8 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 112 x 3 x 416 x 16
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 112 x 3 x 416 x 16

        tx = (-target_x+0.5)*2 # This part is probably responsible for putting the patch in the middle of detection
        ty = (-target_y+0.5)*2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 112 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2

        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg
        # print(adv_batch_t.size()) dim 112 x 3 x 416 x 416
        # print(msk_batch_t.size()) dim 112 x 3 x 416 x 416


        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        # print((adv_batch_t * msk_batch_t).size()) dim = 8 x 14 x 3 x 416 x 416

        return adv_batch_t * msk_batch_t  # It is as if I have passed adv_batch_t "filtered" by the mask itself

# NB output of PatchTransformer is the input of PatchApplier

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):

        advs = torch.unbind(adv_batch, 1)  # Returns a tuple of all slices along a given dimension, already without it.
        # print(np.shape(advs)) # dim = (14,) --> it indicates TODO 14 copies of the adv patch: one for each detected person (random number)
        # plus the remaining to get a total = max_lab (i.e. 14)

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv
            # dim img_batch = 8 x 3 x 416 x 416

            # Put your 14 adv_patches on the image one after the other. When you meet those which are totally 0, i.e. those that do not
            # correspond to a detected object(person) in the image, you keep your image as it is (do nothing). Otherwise, you will have your scaled, rotated and
            # well-positioned patch corresponding to one of the detected objects of the image. I think its pixels are 0s where there is not the object, and =/= 0
            # where there is the object, with appropriate affine properties. Here, you substitute imgage pixels with adv pixels.
            # At the end of the 14th cycle you have attached your patches to all detected regions of the image 'layer by layer', for all images in the batch (8).
        return img_batch


#TODO ____________________________________________________________________________________________________________________________________________________________
    # TODO Summary of PatchTransformer + PatchApplier:
    # take a batch of 6 images, consider one. For it, I have 14 ready adv patches, of which a number that varies for each image is non-zero (remember:
    # the mask is done starting from 0 and 1 labels in lab_batch. Suppose that 5 are non zero. It means that they correspond to 5 detected object in that image.
    # They are already transformed according to correct positions and scales of the 5 detected rectangles. Now, we consider the image of the six composing the batch,
    # and substitute the patches in their positions where they are not zero (so 5 out of 14 in this example)
#TODO ____________________________________________________________________________________________________________________________________________________________


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        #imgsize = 416 from yolo
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self): # magic method
        return self.len

    def __getitem__(self, idx): # magic method
        assert idx <= len(self), 'index range error' # ken(self) refers to __len__(self) method
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')

        # image = self.preprocessing(image)
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5]) # Since the dimension of the label content for yolo is (1*5)

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0) # adds a new dimension to the data shape (5,) => (1,5)

        image, label = self.pad_and_scale(image, label)
        # print(f"Image {type(image)}: {image.size}\n{image}")
        transform = transforms.ToTensor() # transforms the image: [0,255] => floats[0,1] and [H*W*C] => [C*H*W]
        image = transform(image)
        label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        # pad values are '1' for yolo
        return image, label

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

    def pad_and_scale(self, img, lab): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h # modifies the anchor value of the detection box
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            # padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                   # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                   # whose number is not known a priori
        else:
            padded_lab = lab
        #print("Label shape: ", lab.shape)
        #print("Padded label size: ", padded_lab.shape)
        # Label shape:  torch.Size([1, 5])
        # Padded label size:  torch.Size([14, 5])
        return padded_lab



class InriaDataset_2(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, imgsize, shuffle=True):
        #imgsize = 416 from yolo
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        # n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        # assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        #self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        # self.lab_paths = []
        # for img_name in self.img_names:
        #     lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
        #     self.lab_paths.append(lab_path)
        # self.max_n_labels = max_lab

    def __len__(self): # magic method
        return self.len

    def __getitem__(self, idx): # magic method
        assert idx <= len(self), 'index range error' # ken(self) refers to __len__(self) method
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        #lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        # if os.path.getsize(lab_path):       #check to see if label file contains data. 
        #     label = np.loadtxt(lab_path)
        # else:
        #     label = np.ones([5]) # Since the dimension of the label content for yolo is (1*5)

        # label = torch.from_numpy(label).float()
        # if label.dim() == 1:
        #     label = label.unsqueeze(0) # adds a new dimension to the data shape (5,) => (1,5)

        image = self.pad_and_scale(image)
        transform = transforms.ToTensor() # transforms the image: [0,255] => floats[0,1] and [H*W*C] => [C*H*W]
        image = transform(image)
        #label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        # pad values are '1' for yolo
        return image, img_path

    def pad_and_scale(self, img): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                # lab[:, [1]] = (lab[:, [1]] * w + padding) / h # modifies the anchor value of the detection box
                # lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                # lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img

    # def pad_lab(self, lab):
    #     pad_size = self.max_n_labels - lab.shape[0]
    #     if(pad_size>0):
    #         # padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
    #         padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
    #                                                                # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
    #                                                                # whose number is not known a priori
    #     else:
    #         padded_lab = lab
    #     #print("Label shape: ", lab.shape)
    #     #print("Padded label size: ", padded_lab.shape)
    #     # Label shape:  torch.Size([1, 5])
    #     # Padded label size:  torch.Size([14, 5])
    #     return padded_lab