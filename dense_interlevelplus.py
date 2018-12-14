import torch
from math import log, round, sqrt
from collections import OrderedDict

import layers
from operations import group_knn

class Net(torch.nn.Module):
    """3PU inter-level plus skip connection and dense layers"""
    def __init__(self, bradius=1.0,
                 max_up_ratio=16,
                 step_ratio=2, no_res=False,
                 knn=16, growth_rate=12, dense_n=1,
                 fm_knn=3, **kwargs):
        super(Net, self).__init__()
        self.bradius = bradius
        self.max_up_ratio = max_up_ratio
        self.step_ratio = step_ratio
        self.no_res = no_res
        self.knn = knn
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.fm_knn = fm_knn
        num_levels = log(max_up_ratio, step_ratio)
        for l in range(num_levels):
            self.add_module('level_1', Level(dense_n=dense_n, growth_rate=growth_rate, knn=knn))

    def forward(self):
        pass


class Level(torch.nn.Module):
    """3PU per-level network"""
    def __init__(self, dense_n=3, growth_rate=12, knn=16, fm_knn=5, step_ratio=4):
        super(Level, self).__init__()
        self.dense_n = dense_n
        self.fm_knn = fm_knn
        self.step_ratio = step_ratio
        # create code for feature expansion
        if step_ratio < 4:
            # 1x1xstep_ratio
            self.code = self.gen_1d_grid(step_ratio).unsqueeze().detach()
        else:
            expansion_ratio = round(sqrt(step_ratio))**2
            self.code = self.gen_grid(expansion_ratio).unsqueeze().detach()

        in_channels = 3
        self.layer0 = layers.Conv2d(3, 24, [1, 1],  activation=None)
        self.layer1 = layers.DenseEdgeConv(24, growth_rate=growth_rate, n=dense_n, knn=knn)
        in_channels = 84  # 24+(24+growth_rate*dense_n) = 24+(24+36) = 84
        self.layer2_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer2 = layers.DenseEdgeConv(24, growth_rate=growth_rate, n=dense_n, knn=knn)
        in_channels = 144  # 84+(24+36) = 144
        self.layer3_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer3 = layers.DenseEdgeConv(24, growth_rate=growth_rate, n=dense_n, knn=knn)
        in_channels = 204  # 144+(24+36) = 204
        self.layer4_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer4 = layers.DenseEdgeConv(24, growth_rate=growth_rate, n=dense_n, knn=knn)
        in_channels = 264  # 204+(24+36) = 264
        self.up_layer = torch.nn.sequential(OrderedDict([
            ("up_layer1", layers.Conv2d(in_channels+1, 128, 1, activation="relu")),
            ("up_layer2", layers.Conv2d(128, 128, 1, activation="relu")),]))
        self.fc_layer1 = layers.Conv2d(128, 64, 1, activation="relu")
        self.fc_layer2 = layers.Conv2d(64, 3, 1, activation=None)

    def exponential_distance(self, points, knn_points):
        """
        compute knn point distance and interpolation weight for interlevel skip connections
        :param
            points      BxCxN
            knn_points  BxCxNxK
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        if points.dim() == 3:
            points = points.unsqueeze(dim=-1)
        distance = torch.reduce_sum((points - knn_points) ** 2, dim=1, keepdim=True).detach()
        # mean_P(min_K distance)
        h = torch.mean(torch.mean(distance, dim=-1, keepdim=True), dim=-2, keepdim=True)
        weight = torch.exp(-distance / (h/2)).detach()
        return distance, weight

    def gen_grid(self, num_grid_point):
        """
        output [2, num_grid_pointxnum_grid_point]
        """
        x = torch.linspace(-0.2, 0.2, num_grid_point, dtype=torch.float32)
        x, y = torch.meshgrid(x, x)
        grid = torch.stack([x, y], dim=0).view([2, num_grid_point*num_grid_point])  # [2, 2, 2] -> [4, 2]
        return grid

    def gen_1d_grid(self, num_grid_point):
        """
        output [2, num_grid_point]
        """
        grid = torch.linspace(-0.2, 0.2, num_grid_point).view(2, num_grid_point)
        return grid

    def forward(self, x, batch_radius, previous_level4=None):
        # split input into patches
        l0_xyz = x
        x = self.layer0(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        x = torch.cat([self.layer1(x), x], dim=1)
        x = torch.cat([self.layer2(self.layer2_prep(x)), x], dim=1)
        x = torch.cat([self.layer3(self.layer3_prep(x)), x], dim=1)
        x = torch.cat([self.layer4(self.layer4_prep(x)), x], dim=1)
        # interlevel skip connections
        if previous_level4 is not None:
            previous_xyz, previous_feat = previous_level4
            if not self.training and previous_xyz.shape[0] != x.shape[0]:
                previous_xyz = previous_xyz.expand(x.size(0), -1, -1)
                previous_feat = previous_feat.squeeze(dim=-1).expand(x.size(0), -1, -1, )  # BxCxP
            # find closest k point in spatial, BxNxK
            knn_points, knn_idx, knn_distance = group_knn(self.fm_knn, l0_xyz, previous_xyz, unique=True, NCHW=True)
            previous_feat = previous_feat.unsqueeze(2).expand(-1, -1, l0_xyz.size(2))
            knn_idx = knn_idx.unsqueeze(1).expand(-1, previous_feat.size(1), -1, -1)  # BxCxNxK
            knn_feats = torch.gather(previous_feat, 3, knn_idx)
            d, s_average_weight = self.exponential_distance(x, knn_feats)
            d, f_average_weight = self.exponential_distance(l0_xyz, knn_points)
            average_weight = s_average_weight * f_average_weight
            average_weight = average_weight / torch.mean(average_weight+1e-5, dim=-1, keepdim=True)
            knn_feats = torch.mean(average_weight * knn_feats, axis=2, keepdims=True)

            x = 0.2 * knn_feats + x

        ratio = self.code.size(1)
        code = self.code.expand(x.size(0), -1, -1).repeat(x.size(0), ratio, x.size(2))
        code = batch_radius.unsqueeze(-1).unsqueeze(-1) * code
        # BxCxNx1 -> BxCxNxratio -> BxCxNxratio
        # x = torch.reshape(
        #     x.unsqueeze(2).repeat(x.size(0), x.size(1), self.code.size(-1), x.size(3)), [x.shape[0], num_point*expansion_ratio, 1, l4_features.shape[-1]])
