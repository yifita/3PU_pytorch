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

    def set_input(self, input_pc, batch_radius, up_ratio, label_pc=None):
        """
        :param
            input_pc       Bx3xN
            batch_radius   Bx1
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.batch_radius = batch_radius.detach()
        self.up_ratio = up_ratio.detach()
        # gt point cloud
        if label_pc:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        self.predicted, self.batch_radius = self.net(
            self.input, self.batch_radius, ratio=self.up_ratio, gt=self.gt)  # Bx1024 encoded

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

    def save_network(self, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.log_dir, save_filename)
        merge_states = OrderedDict()
        merge_states['states'] = self.net.cpu().state_dict()
        merge_states['opt'] = vars(self.opt)
        torch.save(merge_states, save_path)
        self.net.cuda()

    def load_network(self, path):
        loaded_state = torch.load(path)
        loaded_param_names = set(loaded_state["Net"].keys())
        network = self.net.module if isinstance(
             self.net, torch.nn.DataParallel) else self.net

         # allow loaded states to contain keys that don't exist in current model
         # by trimming these keys;
         own_state = network.state_dict()
          extra = loaded_param_names - set(own_state.keys())
           if len(extra) > 0:
                print('Dropping ' + str(extra) + ' from loaded states')
            for k in extra:
                del loaded_state[name][k]

            try:
                network.load_state_dict(loaded_state[name])
            except KeyError as e:
                print(e)
            print('Loaded {} state from {}'.format(name, path))
