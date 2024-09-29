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

if __name__ == '__main__':

    class PatchTrainer(object):

        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

        def __init__(self, mode):

            self.config = patch_config.patch_configs[mode]()  # select the mode for the patch
            self.model_path = "yolov5n.pt"
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.input_size = 416 # 416 for yolov2 640 for yolov5
            self.names = None # Default COCO
            print(torch.cuda.device_count())

            self.model = YOLOV5TorchObjectDetector(self.model_path, self.device, img_size=(self.input_size, self.input_size),
                                              names=None if self.names is None else self.names.strip().split(","))
            # self.darknet_model = Darknet(self.config.cfgfile_yolov2)
            # self.darknet_model.load_weights(self.config.weightfile_yolov2)

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

                destination_path = "./outputs/Adv/yolo/"
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
                n_epochs = 3
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
                                img.save(os.path.join('./outputs/Adv/yolo/image_steps',name))

                            loss_type = "max_approach"

                            preds, logits = self.model(p_img_batch)
                            # preds -> [[bbox] [class id] [cls names] [confidence]]
                            # logits -> raw unnormalized score before softmax. logits > activation > probability
                            person_confidences = []
                            for class_list, conf_list in zip(preds[2], preds[3]):
                                for cls, conf in zip(class_list, conf_list):
                                    if cls == 'person':
                                        person_confidences.append(conf)
                            conf_mean = sum(person_confidences) / len(person_confidences)
                            print(f"conf_mean: {conf_mean}")
                            score_yolov5 = self.score_extractor_yolo(logits, loss_type)
                            # print("Logits requires grad: ", logits.requires_grad, logits.shape)
                            print(f"Yolo scores (req_grad-{score_yolov5.requires_grad}): {score_yolov5}")

                            nps = self.nps_calculator(adv_patch)
                            tv = self.total_variation(adv_patch)

                            nps_loss = nps * 0.01  # empirically found
                            tv_loss = tv * 2.5  # empirically found

                            # batch operation: mean, max...
                            det_loss = torch.mean(score_yolov5)
                            print("Mean detection loss:, ", det_loss)
                            # det_loss = torch.softmax(det_loss, dim=0)
                            # print("Softmaxed detection loss:, ", det_loss)
                            # print("Length of TrainLoader: ", len(train_loader))


                            if use_cuda:
                                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                                # loss = det_loss
                            else:
                                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1))
                                # loss = det_loss

                            ep_det_loss += det_loss.detach().cpu().numpy() / len(
                                train_loader)  # Normalizing to keep the value checked from increasing sequentially
                            ep_nps_loss += nps_loss.detach().cpu().numpy()
                            ep_tv_loss += tv_loss.detach().cpu().numpy()
                            ep_loss += loss

                            ep_det_loss_raw += conf_mean/ len(train_loader)#torch.sigmoid(det_loss).detach().cpu().numpy() # For graph
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

                            if i_batch + 1 >= len(train_loader):
                                print('\n')
                            else:
                                del adv_batch_t, preds, logits, score_yolov5, det_loss, p_img_batch, nps_loss, tv_loss, loss

                                if use_cuda:
                                    torch.cuda.empty_cache()

                            p_img_batch = p_img_batch.detach()
                            score_yolov5 = score_yolov5.detach()
                            # Grad check
                            print("p_img_batch_grad: ", p_img_batch.requires_grad)
                            # print("preds_grad: ", preds.requires_grad)
                            # print("logits_grad: ", logits.requires_grad)
                            # print("person_conf_grad: ", person_confidences.requires_grad)
                            print("score_tolov5", score_yolov5.requires_grad)

                            print("Losses-", 'det:', det_loss.requires_grad, 'nps:', nps_loss.requires_grad,
                                  'tv:', tv_loss.requires_grad)
                            print("patch_grad: ", adv_patch_cpu.requires_grad)

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
                        step_name = 'step_' + str(tot_cnt) + '.jpg'
                        step_folder = os.path.join("./outputs/Adv/yolo/patch_steps", step_name)
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
                        im.save(os.path.join("./outputs/Adv/yolo", im_name))

                        del adv_batch_t, preds, logits, score_yolov5, det_loss, p_img_batch, nps_loss, tv_loss, loss

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
            step_folder = os.path.join("./outputs/Adv/yolo/patch_steps", step_name)
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


    use_cuda = 0
    mode = 'paper_obj'
    trainer = PatchTrainer(mode)
    trainer.train()
    plot_sep_loss()
