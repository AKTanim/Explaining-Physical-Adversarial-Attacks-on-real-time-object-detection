from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from utils_yolov2.util import *
import argparse
import os 
import os.path as osp
from utils_yolov2.darknet import Darknet
from utils_yolov2.preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl

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
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("utils_yolov2/imgs/dog.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v2 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "..\images\inria\\adv\ocl_images", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "..\outputs\cam_test\yolo_v2_final", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "coco")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)

    return parser.parse_args()


if __name__ ==  '__main__':
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    
    if args.dataset == "pascal":
        inp_dim = 416
        num_classes = 20
        classes = load_classes('utils_yolov2/data/voc.names')
        weightsfile = 'utils_yolov2/weights/yolo-voc.weights'
        cfgfile = "utils_yolov2/cfg/yolo-voc.cfg"

    
    elif args.dataset == "coco":
        inp_dim = 544
        num_classes = 80
        classes = load_classes('utils_yolov2/data/coco.names')
        weightsfile = 'utils_yolov2/weights/yolov2.weights'
        cfgfile = "utils_yolov2/cfg/yolo.cfg"
        
    else: 
        print("Invalid dataset")
        exit()

        
    stride = 32

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")
    
    
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    # model(get_test_input(inp_dim, CUDA))
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
        print('imlist: ',imlist)
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    load_batch = time.time()
    

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        


    i = 0
    
    output = torch.FloatTensor(1, 8)
    write = False
#    model(get_test_input(inp_dim, CUDA))
    
    start_det_loop = time.time()
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
       
        prediction = model(Variable(batch, volatile = True))
        
        prediction = prediction.data 
        
        
        
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        
        prediction = predict_transform(prediction, inp_dim, stride, model.anchors, num_classes, confidence, CUDA)
        print('Prediction:', prediction.shape)
        
            
        if type(prediction) == int:
            i += 1
            continue
        
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, num_classes, nms = True, nms_conf = nms_thesh)
        prediction = prediction.view(-1, 8)
        print('Prediction_2:', prediction.shape, prediction)
        
        
        end = time.time()
        
                    
#        print(end - start)

            

        prediction[:,0] += i*batch_size
        
    
            
        
          
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        # print('Output:', output.shape, output)
            

        for image in imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]:
            im_id = imlist.index(image)
            print('im_id', im_id)
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("\\")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        
        if CUDA:
            torch.cuda.synchronize()
    
    
    output_recast = time.time()
    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())/inp_dim
    output[:,1:5] *= im_dim_list
    
    
    class_load = time.time()

    colors = pkl.load(open("utils_yolov2/pallete", "rb"))
    
    
    draw = time.time()


    def write(x, batches, results):
        c1 = (x[1].int().item(), x[2].int().item())
        c2 = (x[3].int().item(), x[4].int().item())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}, conf:{1:.2f}".format(classes[cls], x[5].item())
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
    
            
    list(map(lambda x: write(x, im_batches, orig_ims), output))
      
    det_names = pd.Series(imlist).apply(lambda x: "{}\det_{}".format(args.det,x.split("\\")[-1]))
    print('det_names:', det_names)
    list(map(cv2.imwrite, det_names, orig_ims))
    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    
    torch.cuda.empty_cache()
        
        
        
        
    
    
