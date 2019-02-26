from collections import OrderedDict
import os
import torch
from math import log
from collections import defaultdict

from network.model_loss import ChamferLoss
from utils.pytorch_utils import load_network, save_network


class Model(object):
    def __init__(self, net, phase, opt):
        self.net = net
        self.phase = phase

        if phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                              betas=(0.9, 0.999))

        if opt.ckpt is not None:
            self.step = load_network(self.net, opt.ckpt)
        else:
            self.step = 0

    def set_input(self, input_pc, up_ratio, label_pc=None):
        """
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.up_ratio = up_ratio
        # gt point cloud
        if label_pc is not None:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt = self.net(
                self.input, ratio=self.up_ratio, gt=self.gt)
        else:
            self.predicted = self.net(
                self.input, ratio=self.up_ratio)

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
        loss_chamfer = self.chamfer_criteria(
            pc.transpose(1, 2).contiguous(),
            pc_label.transpose(1, 2).contiguous())
        weight = log(self.net.max_up_ratio / self.up_ratio, self.net.step_ratio)
        loss_chamfer = loss_chamfer * weight
        prev_err = self.error_log["cd_loss_x{}".format(self.up_ratio)]
        self.error_log["cd_loss_x{}".format(
            self.up_ratio)] = prev_err + (loss_chamfer.item() - prev_err) / (self.step + 1)
        return loss_chamfer

    def test_model(self):
        self.net.eval()
        self.forward()
