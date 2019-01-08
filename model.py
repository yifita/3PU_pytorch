from collections import OrderedDict
import os
import torch

from model_loss import ChamferLoss
from utils.pytorch_utils import load_network, save_network


class Model(object):
    def __init__(self, net, phase, opt):
        self.net = net

        if phase == 'train':
            self.chamfer_criteria = ChamferLoss()
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=self.opt.lr_init,
                                              betas=(0.9, 0.999))

        if opt.ckpt is not None:
            self.step = load_network(opt.ckpt)

    def set_input(self, input_pc, up_ratio, label_pc=None):
        """
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.up_ratio = up_ratio.detach()
        self.radius = radius
        # gt point cloud
        if label_pc:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        if phase == "train":
            self.predicted, self.gt = self.net(
                self.input, ratio=self.up_ratio, gt=self.gt)  # Bx1024 encoded
        else:
            self.predicted = self.net(
                self.input, ratio=self.up_ratio, gt=self.gt)  # Bx1024 encoded

    def optimize(self, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()

        self.net.train()
        self.forward()

        loss = self.compute_chamfer_loss(self.predicted, self.gt)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 1)
        self.optimizer.step()
        self.step += 1

    def compute_chamfer_loss(self, pc, pc_label):
        loss_chamfer = self.chamfer_criteria(pc_label.transpose(
            1, 2).contiguous(), pc.transpose(1, 2).contiguous())
        up_ratio = self.up_ratio.item()
        return loss_chamfer

    def test_model(self):
        self.net.eval()
        self.forward()
