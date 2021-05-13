from __future__ import absolute_import, division, print_function

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

        self.models["unet"] = networks.UNet(n_channels=1, n_classes=4)

        self.models["unet"].to(self.device)
        self.parameters_to_train += list(self.models["unet"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train,
                                          self.opt.learning_rate)

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
