from collections import OrderedDict
import os
import torch

from model_loss import ChamferLoss


class Model(object):
    def __init__(self, net, phase, opt):
        self.opt = opt
        self.net = net

        if phase == 'train':
            self.chamfer_criteria = ChamferLoss(threshold=opt.CD_threshold)
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=self.opt.lr_init,
                                              betas=(0.9, 0.999))

        if opt.ckpt is not None:
            self.load_network(opt.ckpt)
            print(" Previous weight loaded {}".format(opt.resume))

    def set_input(self, input_pc, up_ratio, label_pc=None):
        """
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.up_ratio = up_ratio.detach()
        # gt point cloud
        if label_pc:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        self.predicted, self.batch_radius = self.net(
            self.input, ratio=self.up_ratio, gt=self.gt)

    def optimize(self, epoch=None):
        self.optimizer.zero_grad()

        self.net.train()
        self.forward()

        loss = self.compute_chamfer_loss(self.predicted, self.gt)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 1)
        self.optimizer.step()

    def update_progress(self):
        pass

    def compute_chamfer_loss(self, pc, pc_label):
        loss_chamfer = self.chamfer_criteria(pc_label.transpose(
            1, 2).contiguous(), pc.transpose(1, 2).contiguous())
        up_ratio = self.up_ratio.item()
        return loss_chamfer

    def test_model(self):
        self.net.eval()
        self.forward()
