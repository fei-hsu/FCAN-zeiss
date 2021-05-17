from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import json
import os
import networks
import datasets
from torchvision import transforms
from train_utils import *
from eval import EvalMetrics

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from networks.aan_net import AAN_Loss

from torchvision.utils import save_image

from torch.autograd import Variable

from networks import DeepLab_ResNet101_MSC
from networks.ran_net import RAN

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

from networks.pyramid_pooling import SpatialPyramidPooling

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/resnet_pretrained"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet50"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def grey_to_rgb(img):
    img = torch.cat((img, img, img), 0)
    return img

def rescale_transform(img, out_range=(0, 1)):
    img = img - img.min()
    img /= (img.max() - img.min())
    img *= (out_range[1] - out_range[0])
    return img


activation = {}
def get_activation(name):
    def hook(model, input, output):
        # activation[name] = output.detach()
        activation[name] = output
    return hook

CIRRUS = "/home/zeju/Documents/zeiss_domain_adaption/splits/cirrus_samples.txt"
SPECTRALIS = "/home/zeju/Documents/zeiss_domain_adaption/splits/cirrus_samples.txt"
CIRRUS_SAMPLE = "/home/zeju/Documents/zeiss_domain_adaption/Retouch-dataset/pre_processed/Cirrus_part1/3c68f67cd2e2b41afa54bf6059f509d1/image/039.jpg"
SPECTRALIS_SAMPLE = "/home/zeju/Documents/zeiss_domain_adaption/Retouch-dataset/pre_processed/Spectralis_part1/7b2607e057592d507c4ec4732bae64c2/image/015.jpg"

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # self.models["encoder"] = networks.ResnetEncoder(
        #     self.opt.num_layers, pretrained=False)
        # self.models["encoder"].to(self.device)
        # self.parameters_to_train += list(self.models["encoder"].parameters())
        #
        # self.models["decoder"] = networks.Decoder(
        #     self.models["encoder"].num_ch_enc)
        # self.models["decoder"].to(self.device)
        # self.parameters_to_train += list(self.models["decoder"].parameters())

        # Initialize the resnet50 and resnet101 model for this run
        model_50, input_size = self.initialize_model("resnet50", num_classes, feature_extract, use_pretrained=True)
        self.models["resnet50"] = model_50
        self.models["resnet50"].to(self.device)

        model_101, input_size = self.initialize_model("resnet101", num_classes, feature_extract, use_pretrained=True)
        self.models["resnet101"] = model_101
        self.models["resnet101"].to(self.device)

        # self.models["RAN"] = DeepLab_ResNet101_MSC(n_classes=21)
        self.models["RAN"] = RAN(in_channels=2048, out_channels=128)
        self.models["RAN"].to(self.device)
        
        self.models["unet"] = networks.UNet(n_channels=1, n_classes=4)
        self.models["unet"].to(self.device)


        # self.models["unet"] = networks.UNet(n_channels=1, n_classes=4)
        # self.models["unet"].to(self.device)

        # self.parameters_to_train += list(self.models["unet"].parameters())
        # self.parameters_to_train += list(self.models["resnet50"].parameters())
        # self.parameters_to_train += list(self.models["resnet101"].parameters())

        self.parameters_to_train = nn.Parameter(rescale_transform(torch.normal(mean=0.5, std=1, size=(1, 3, 512, 512), device="cuda")), requires_grad=True)

        '''
        w = Variable(torch.randn(3, 5), requires_grad=True)
        b = Variable(torch.randn(3, 5), requires_grad=True)
        self.parameters_to_train += w
        self.parameters_to_train += b
        '''

        #self.model_optimizer = optim.SGD(self.parameters_to_train,self.opt.learning_rate,momentum=0.9,weight_decay=0.0005)
        self.model_optimizer = optim.Adam([self.parameters_to_train],
                                          self.opt.learning_rate)

        '''
        self.model_optimizer = optim.Adam(self.parameters_to_train,
                                          self.opt.learning_rate)
        '''

        self.dataset = datasets.Retouch_dataset

        if self.opt.use_augmentation:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 #transforms.RandomRotation(degrees=(-20, 20)),
                                                 ])
        else:
            self.transform = None

        # self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.opt.ce_weighting).to(self.device),
        #                                      ignore_index=self.opt.ignore_idx)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        train_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='train',
            is_train=True,
            transform=self.transform)

        train_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='train',
            is_train=True,
            transform=self.transform)


        train_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='train',
            is_train=True,
            transform=self.transform)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        val_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='val',
            is_train=False,
            transform=self.transform)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    '''
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.opt.save_model:
                self.save_model()
    '''

    def train(self):
        input = rescale_transform(torch.normal(mean=0.5, std=1, size=(1, 3, 512, 512), device="cuda"))
        input.requires_grad_(True)

        save_image(input, 'img.png')

        I = 1000
        # L_RAN = self.compute_RAN_loss(x_o, x_t)
        # print("ran loss", L_RAN.shape)

        '''
        source_img = grey_to_rgb(transforms.ToTensor()(pil_loader(CIRRUS_SAMPLE))).unsqueeze(0).cuda()
        source_img.requires_grad_(False)
        target_img = grey_to_rgb(transforms.ToTensor()(pil_loader(SPECTRALIS_SAMPLE))).unsqueeze(0).cuda()
        target_img.requires_grad_(False)
        '''
        source_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(CIRRUS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        target_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(SPECTRALIS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        save_image(source_img, 'source_img.png')
        save_image(target_img, 'target_img.png')

        inputs = {}
        inputs["source"] = source_img
        inputs["target"] = target_img
        L_RAN = self.compute_RAN_loss(inputs)

        # print("base_dir", self.opt.base_dir) /home/zeju/Documents/zeiss_domain_adaption/Retouch-dataset/pre_processed
        # print("list_dir", self.opt.list_dir) /home/zeju/Documents/zeiss_domain_adaption/splits/split_cirrus

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.step in range(I):
            before_op_time = time.time()

            '''
            L_ANN = AAN_Loss()
            loss = L_ANN(input, source_img, target_img)
            loss.backward()
            # loss = torch.autograd.Variable(L_ANN.forward(), requires_grad=True)
            print("loss", loss)
            # grad = torch.autograd.grad(loss, input, allow_unused=True)
            print("grad", input.grad)
            '''

            w = 1000 * (I - self.step) / I
            L_ANN = self.compute_AAN_loss(input, source_img, target_img)
            L_ANN.backward()

            # print("input", input)
            # print("L_ANN", L_ANN)
            # print("input max", input.max())
            # print("input grad max", input.grad.max())
            test = w * input.grad / torch.linalg.norm(input.grad.view(-1), ord=1)
            # print("test max", test.max())
            input = input.data - w * input.grad / torch.linalg.norm(input.grad.view(-1), ord=1)
            input.requires_grad_(True)

            # self.model_optimizer.zero_grad()
            # self.model_optimizer.step()

            duration = time.time() - before_op_time

            print("step", self.step, "loss", L_ANN, "duration", duration)

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 #and self.step < 2000
            # late_phase = self.step % 2000 == 0

            '''
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                print("train", inputs, outputs, losses)
                self.val()
            '''

            self.step += 1
        img = input[0]
        save_image(img, 'img.png')

    def test(self, x):
        out = x.sum()
        return out


    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 #and self.step < 2000
            # late_phase = self.step % 2000 == 0

            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key != 'case_name':
                inputs[key] = ipt.to(self.device)

        outputs = {}

        # features = self.models["encoder"](inputs["image"])
        # preds = self.models["decoder"](features)
        preds = self.models["unet"](inputs["image"])

        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.compute_accuracy(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)

            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        pred = outputs['pred']
        target = inputs['label']
        # print(pred[0,,2,3])
        ce_loss = self.criterion(pred,
                                 target)
        mask = torch.zeros_like(ce_loss)

        mask_idx = (outputs["pred_idx"] > 0).squeeze(1)

        if mask_idx.sum() > 100:
            mask[mask_idx] = 1
        else:
            mask[(target > 0)] = 1

        outputs["mask"] = mask.unsqueeze(1)
        to_optimise = (ce_loss * mask).sum() / (mask.sum() + 1e-5)

        total_loss += to_optimise
        losses["loss"] = total_loss
        return losses

    def compute_AAN_loss(self, input, source, target):
        M_o = []
        M_s = []
        G_o = []
        G_t = []
        L = ['conv1', 'res2c', 'res3d', 'res4f', 'res5c']
        # model_ft.layer1[0].conv2.register_forward_hook(hook_fn)
        self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet50"](input)

        for layer in L:
            M_o.append(activation[layer])
            G_o.append(self.generate_style_image(activation[layer]))
            del activation[layer]

        self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet50"](source)

        for layer in L:
            M_s.append(activation[layer])
            # print("activation layer", layer, activation[layer])
            del activation[layer]

        self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet50"](target)

        for layer in L:
            # G_t.append(activation[layer])
            G_t.append(self.generate_style_image(activation[layer]))
            del activation[layer]

        alpha = 1e-14
        loss = torch.tensor(0, device=self.device)
        for i in range(len(M_o)):
            loss = loss + torch.dist(M_o[i], M_s[i], 2) + alpha * torch.dist(G_o[i], G_t[i], 2)

        # test_loss = Variable(loss, requires_grad=True)
        return loss


    def compute_RAN_loss(self, inputs):
        print("compute RAN loss")
        features = {}

        self.models["resnet101"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet101"](inputs["source"])
        features["source"] = self.models['RAN'](activation['res5c'])
   
        
        outputs = {}
        pooling = SpatialPyramidPooling(levels=[1])
        p = pooling(activation['res5c'])
        L_seg = 0
        '''
        preds = self.models["unet"](p)
        print("preds shape", preds.shape)
        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)
        target = inputs['label']

        L_seg = self.compute_losses(inputs, outputs)
        print("L_seg", L_seg)
        '''
        del activation['res5c']
        output = self.models["resnet101"](inputs["target"])
        features["target"] = self.models['RAN'](activation['res5c'])
        #print("features target shape", features["target"].shape)



        #devide input into target domain or dource domain
        L_adv = - torch.mean(torch.log(features["target"])) - torch.mean(torch.log(1-features["source"]))
        print("L_adv", L_adv)
        
        loss = L_adv - 5*L_seg
        # print("test shape", test.shape)

        return loss

    def generate_style_image(self, inputs):
        c = inputs.shape[1]
        M_i = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2] * inputs.shape[3]).to(self.device)
        M_j = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2] * inputs.shape[3]).permute(0, 2, 1).to(self.device)
        style_image = torch.bmm(M_i, M_j)
        '''
        style_image1 = torch.zeros(c, c).to(self.device)
        for i in range(c):
            for j in range(c):
                a = torch.flatten(inputs[:, i, :, :])
                b = torch.flatten(inputs[:, j, :, :])
                style_image1[i, j] = torch.dot(a, b)
        test = style_image - style_image1
        print("style image sum", style_image.sum())
        print("test sum", test.sum())
        '''
        return style_image


    def compute_accuracy(self, inputs, outputs, losses):
        evaluation = EvalMetrics(outputs["pred_idx"],
                                 inputs['label'],
                                 n_classes=4)

        losses["accuracy/dice"] = evaluation.dice_coef()
        losses["accuracy/iou"] = evaluation.iou_coef()


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image("inputs/{}".format(j), inputs["image"][j].data, self.step)
            writer.add_image("labels/{}".format(j), normalize_image(inputs["label"][j].unsqueeze(0).data), self.step)
            writer.add_image("predictions/{}".format(j), normalize_image(outputs["pred_idx"][j].data), self.step)
            writer.add_image("positive_region/{}".format(j), outputs["mask"][j].data, self.step)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet50":
            """ Resnet50
            """
            print("loading resnet50")
            # model_ft = models.resnet50(pretrained=use_pretrained)
            model_ft = models.resnet50()
            model_ft.load_state_dict(torch.load('model/pretrained/resnet50.pth'))
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet101":
            """ Resnet101
            """
            print("loading resnet101")
            # model_ft = models.resnet101(pretrained=use_pretrained)
            model_ft = models.resnet101()
            model_ft.load_state_dict(torch.load('model/pretrained/resnet101.pth'))
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history





