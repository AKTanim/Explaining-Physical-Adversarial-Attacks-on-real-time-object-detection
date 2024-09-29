"""
Training code for Adversarial patch training on YOLO v5.
"""

import PIL
import torch
from tqdm import tqdm
import statistics

from utils_adv.load_data import *
from torch import autograd
from torchvision import transforms
from utils_adv.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from utils_adv.plots import *
from utils_adv import patch_config
import time
import os

from inference.utils_rtdetr.rtdetr import load_model
# from inference.rtdetr_inference import manage_score

if __name__ == '__main__':

    class PatchTrainer(object):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

        def __init__(self, mode):

            self.config = patch_config.patch_configs[mode]()  # select the mode for the patch
            self.model_name = "rtdetr_resnet50"
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.input_size = 640 # 416 for yolov2 640 for yolov5 and rtdetr
            self.names = None # Default COCO
            print(torch.cuda.device_count())

            self.model = load_model(self.model_name)


            if use_cuda:
                self.model = self.model.eval().to(self.device)
                self.patch_applier = PatchApplier().to(self.device)
                self.patch_transformer = PatchTransformer().to(self.device)
                self.score_extractor_yolo = yolov5_feature_output_manage(0, 80, self.config).to(self.device)
                self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).to(self.device)
                self.total_variation = TotalVariation().to(self.device)
            else:
                self.model = self.model.eval()
                self.patch_applier = PatchApplier()
                self.patch_transformer = PatchTransformer()
                self.score_extractor_yolo = yolov5_feature_output_manage(0, 80, self.config)
                # self.score_extractor_yolo = yolov2_feature_output_manage(0, 80, self.config)
                self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)
                self.total_variation = TotalVariation()

            self.batch_arr = [*range(8, 9, 2)]

        def train(self):

            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """

            for batch_size in self.batch_arr:

                img_size = self.input_size

                destination_path = "./outputs/Adv/rtdetr/"
                destination_name = 'loss_tracking_yv2_obj_noepoch.txt'
                destination_name2 = 'loss_tracking_compact_batch_yv2_obj_noepoch.txt'
                destination_name3 = 'loss_over_epochs.txt'
                dest_4_name = 'loss_over_batch_' + str(batch_size) + '.txt'
                destination = os.path.join(destination_path, destination_name)
                destination2 = os.path.join(destination_path, destination_name2)
                destination3 = os.path.join(destination_path, destination_name3)
                dest_4 = os.path.join(destination_path, dest_4_name)
                textfile = open(destination, 'w+')
                textfile2 = open(destination2, 'w+')
                textfile3 = open(destination3, 'w+')
                textfile4 = open(dest_4, 'w+')

                # batch_size = self.config.batch_size
                n_epochs = 10
                max_lab = 14  # max 14 person can be detected in an image

                # Generate starting point
                adv_patch_cpu = self.generate_patch("gray")
                # adv_patch_cpu = self.read_image("images/eagle.jpg")

                adv_patch_cpu.requires_grad_(True)  # allows adaptation of the image after each batch

                train_loader = torch.utils.data.DataLoader(
                    InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                 shuffle=True),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=10)
                self.epoch_length = len(train_loader)
                print(f'One epoch is {len(train_loader)}')

                optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate,
                                       amsgrad=True)  # starting lr = 0.03
                scheduler = self.config.scheduler_factory(optimizer)

                tot_cnt = 0
                et0 = time.time()  # epoch start
                for epoch in range(n_epochs):
                    print('-' * 80, f'Epoch **{epoch}** starts:')
                    ep_det_loss = 0
                    ep_nps_loss = 0
                    ep_tv_loss = 0
                    ep_loss = 0
                    ep_det_loss_raw = 0
                    bt0 = time.time()  # batch start
                    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                                total=self.epoch_length):
                        with autograd.detect_anomaly():

                            if use_cuda:
                                img_batch = img_batch.to(self.device)
                                lab_batch = lab_batch.to(self.device)
                                adv_patch = adv_patch_cpu.to(self.device)
                            else:
                                img_batch = img_batch
                                lab_batch = lab_batch
                                adv_patch = adv_patch_cpu

                            adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True,
                                                                 rand_loc=False)
                            p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                            p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                            print("Image batch after patch application: ", len(p_img_batch))
                            for idx, img in enumerate(p_img_batch):
                                img = transforms.ToPILImage('RGB')(img)
                                name = 'ep_'+str(epoch)+'_bch_'+str(i_batch)+'_'+str(idx)+'.png'
                                img.save(os.path.join('./outputs/Adv/rtdetr/image_steps',name))

                            loss_type = "max_approach"

                            outputs = self.model(p_img_batch)
                            print('before manage score:', outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
                            print("Main output:", outputs.keys())
                            logits, confidences = self.manage_score(outputs)
                            print(f"Score after: {logits.shape}, {type(logits)}, {logits}\n"
                                  f"{type(confidences), confidences}")
                            # preds, logits = outputs['pred_boxes'], outputs['pred_logits']
                            # preds, logits = self.non_max_suppression(preds, logits)
                            # preds -> [[bbox] [class id] [cls names] [confidence]]
                            # logits -> raw unnormalized score before softmax. logits > activation > probability
                            # person_confidences = []
                            # for class_list, conf_list in zip(preds[2], preds[3]):
                            #     for cls, conf in zip(class_list, conf_list):
                            #         if cls == 'person':
                            #             person_confidences.append(conf)
                            conf_mean = sum(confidences) / len(confidences)
                            print(f"conf_mean: {conf_mean}")
                            # score_yolov5 = self.score_extractor_yolo(logits, loss_type)
                            score_rtdetr = logits[:, 0] # 0 = 'person'
                            print("Logits requires grad: ", logits.requires_grad, logits.shape)
                            print(f"rtDETR scores (req_grad-{score_rtdetr.requires_grad}): {score_rtdetr}")

                            nps = self.nps_calculator(adv_patch)
                            tv = self.total_variation(adv_patch)

                            nps_loss = nps #* 0.01  # empirically found
                            tv_loss = tv #* 2.5  # empirically found

                            # batch operation: mean, max...
                            det_loss = torch.mean(score_rtdetr)/10
                            print("Mean detection loss:, ", det_loss)
                            # det_loss = torch.softmax(det_loss, dim=0)
                            # print("Softmaxed detection loss:, ", det_loss)
                            # print("Length of TrainLoader: ", len(train_loader))


                            if use_cuda:
                                loss = det_loss + nps_loss + tv_loss#torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                                # loss = det_loss
                            else:
                                loss = det_loss + nps_loss + tv_loss#torch.max(tv_loss, torch.tensor(0.1))
                                # loss = det_loss

                            ep_det_loss += det_loss.detach().cpu().numpy() / len(
                                train_loader)  # Normalizing to keep the value checked from increasing sequentially
                            ep_nps_loss += nps_loss.detach().cpu().numpy()
                            ep_tv_loss += tv_loss.detach().cpu().numpy()
                            ep_loss += loss

                            ep_det_loss_raw += det_loss/ len(train_loader)#torch.sigmoid(det_loss).detach().cpu().numpy() # For graph
                            # print("ep_det_loss_raw_inside batch: ", ep_det_loss_raw)

                            # Important: Gradient of loss is necessary to compute the backward pass.
                            # Hence converting anything to list or numpy related to loss will prevent pytorch
                            # from calculating the gradients properly.
                            # Optimization step + backward
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                            bt1 = time.time()  # batch end
                            if i_batch % 1 == 0:
                                textfile.write(
                                    f'i_batch: {i_batch}\nb_tot_loss:{loss}\nb_det_loss: {det_loss}\nb_nps_loss: {nps_loss}\nb_TV_loss: {tv_loss}\n\n')
                                textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')
                                textfile4.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')

                            tot_cnt += 1
                            p_img_batch = p_img_batch.detach()
                            score_rtdetr = score_rtdetr.detach()
                            # Grad check
                            print("p_img_batch_grad: ", p_img_batch.requires_grad)
                            # print("preds_grad: ", preds.requires_grad)
                            # print("logits_grad: ", logits.requires_grad)
                            # print("person_conf_grad: ", person_confidences.requires_grad)
                            print("score_tolov5", score_rtdetr.requires_grad)

                            print("Losses-", 'det:', det_loss.requires_grad, 'nps:', nps_loss.requires_grad,
                                  'tv:', tv_loss.requires_grad)
                            print("patch_grad: ", adv_patch_cpu.requires_grad)

                            if i_batch + 1 >= len(train_loader):
                                print('\n')
                            else:
                                del adv_batch_t, outputs, logits, score_rtdetr, det_loss, p_img_batch, nps_loss, tv_loss, loss

                                if use_cuda:
                                    torch.cuda.empty_cache()

                            bt0 = time.time()

                    print('ep_img_batch_grad: ', p_img_batch.requires_grad)
                    et1 = time.time()  # epoch end

                    ep_det_loss = ep_det_loss / len(train_loader)  # Normalize again to match with previous normalization inside the batch loop
                    ep_nps_loss = ep_nps_loss / len(train_loader)  # Normalize other losses only once
                    ep_tv_loss = ep_tv_loss / len(train_loader)
                    ep_loss = ep_loss / len(train_loader)

                    ep_det_loss_raw = ep_det_loss_raw / len(train_loader)
                    # print("ep_det_loss_raw_after epoch: ", ep_det_loss_raw)

                    scheduler.step(ep_loss)  # adjusts the learning rate based on ep_loss

                    if True:
                        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                        step_name = 'step_' + str(epoch) + '.jpg'
                        step_folder = os.path.join("./outputs/Adv/rtdetr/patch_steps", step_name)
                        im.save(step_folder)

                        print('  EPOCH NR: ', epoch),
                        print('EPOCH LOSS: ', ep_loss)
                        print('  DET LOSS: ', ep_det_loss_raw)
                        print('  NPS LOSS: ', ep_nps_loss)
                        print('   TV LOSS: ', ep_tv_loss)
                        print('EPOCH TIME: ', et1 - et0)

                        textfile.write(
                            f'\ni_epoch: {epoch}\ne_total_loss:{ep_loss}\ne_det_loss: {ep_det_loss}\ne_nps_loss: {ep_nps_loss}\ne_TV_loss: {ep_tv_loss}\n\n')
                        # textfile3.write(f'{epoch} {ep_loss} {ep_det_loss} {ep_nps_loss} {ep_tv_loss}\n')
                        textfile3.write(f'{epoch} {ep_loss} {ep_det_loss_raw} {ep_nps_loss} {ep_tv_loss}\n')

                        # Plot the final adv_patch (learned) and save it
                        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                        im_name = str(mode) + "_" + str(n_epochs) + "ep_" + str(batch_size) + "bch" + ".jpg"
                        im.save(os.path.join("./outputs/Adv/rtdetr", im_name))

                        del adv_batch_t, outputs, logits, score_rtdetr, det_loss, p_img_batch, nps_loss, tv_loss, loss

                        if use_cuda:
                            torch.cuda.empty_cache()

                    et0 = time.time()


        def generate_patch(self, type):
            """
            Generate a random patch as a starting point for optimization.

            :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
            :return:
            """
            if type == 'gray':
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
                # Returns a tensor of specific size filled with the value 0.5 (gray color on the interval [0,1])
            elif type == 'random':
                adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
                # Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1]

            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            step_name = 'step_' + '0' + '.jpg'
            step_folder = os.path.join("./outputs/Adv/rtdetr/patch_steps", step_name)
            im.save(step_folder)

            return adv_patch_cpu

        def read_image(self, path):
            """
            Read an input image to be used as a patch

            :param path: Path to the image to be read.
            :return: Returns the transformed patch as a pytorch Tensor.
            """
            patch_img = Image.open(path).convert('RGB')
            tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()

            adv_patch_cpu = tf(patch_img)
            return adv_patch_cpu

        def manage_score(self, outputs):
            logits = []
            logit = outputs['pred_logits']
            cls_id = 0 # 'person' class
            confidences = []
            # print('Detected objects:', len(outputs['pred_boxes']))

            soft_logit = torch.softmax(logit, dim=-1)  # probability
            # print('soft_logit:', soft_logit.shape)
            cls_logit = soft_logit[:, :, cls_id]
            # print('cls_logit: ', cls_logit.shape)
            # filters out low probability 'person' detections out of 300 detections
            max_index = torch.argwhere(cls_logit > 0.5)
            print('max_idx:', max_index.shape, max_index)
            for idx in max_index:
                i, j = idx
                # print('i,j: ', i, j)
                logits.append(logit[i.item(), j.item(), :]) # i,j,80

                confidence = cls_logit[i.item(), j.item()].detach().numpy().tolist()
                confidence = round(confidence, 2)  # or "{:.2f}".format(confidence)
                confidences.append(confidence)

            logits_tensor = torch.stack(logits)
            print('logits: ', logits_tensor.shape)
            return logits_tensor, confidences

        def non_max_suppression(self, prediction, logits, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False,
                                multi_label=False, labels=(), max_det=300):
            """Runs Non-Maximum Suppression (NMS) on inference and logits results

            Returns:
                 list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
            """

            nc = prediction.shape[-1]  # number of classes
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


    use_cuda = 0
    mode = 'paper_obj'
    trainer = PatchTrainer(mode)
    trainer.train()
    plot_sep_loss()
