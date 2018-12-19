import torch
from math import log, round, sqrt
from collections import OrderedDict

import layers
import operations


class Net(torch.nn.Module):
    """3PU inter-level plus skip connection and dense layers"""

    def __init__(self, bradius=1.0,
                 max_up_ratio=16,
                 step_ratio=2, no_res=False,
                 knn=16, growth_rate=12, dense_n=3,
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
            self.add_module('level_%d' % l, Level(
                dense_n=dense_n, growth_rate=growth_rate, knn=knn))

    def extract_xyz_feature_patch(self, batch_xyz, k, batch_features=None, gt_xyz=None, gt_k=None):
        """
        extract a single patch via from input xyz and also return their corresponding features
        :param
            batch_xyz: Bx3xN
            k patch:   size
            batch_features: BxCxN
            gt_xyz:    Bx3xM
            gt_k:      size of ground truth patch
        """
        batch_size, _, num_point = batch_xyz.shape.as_list()
        if self.training:
            patch_num = 1
            seed_idx = torch.randint(low=0, high=num_point, size=[batch_size, patch_num], dtype=torch.int32,
                                     layout=torch.strided, device=batch_xyz.device)
        else:
            assert(batch_size == 1)
            # remove residual,
            _, _, closest_d = operations.group_knn(
                2, batch_xyz, batch_xyz, unique=False)
            # BxN
            closest_d = closest_d[:, :, 1]
            # BxN, points whose NN is within a threshold Bx1
            mask = closest_d < 5*(torch.mean(closest_d, dim=1, keepdim=True))
            # Bx1xN
            mask = torch.unsqueeze(mask, dim=1).expand_as(batch_xyz)
            # filter (B, P', 3)
            batch_xyz = torch.masked_select(
                batch_xyz, mask).view(batch_size, 3, num_point)
            patch_num = int(num_point / k * 3)
            batch_xyz_transposed = batch_xyz.transpose(2, 1).contiguous()
            idx = operations.furthest_point_sample(
                batch_xyz_transposed, patch_num)
            batch_seed_point = operations.gather_points(batch_xyz, idx)
            k = torch.min([k, num_point])

        # Bx3xM M=patch_num
        batch_seed_point = operations.gather_points(batch_xyz, seed_idx)
        # Bx3xMxK, BxMxK
        batch_xyz, new_patch_idx, _ = operations.group_knn(
            k, batch_xyz, batch_seed_point, unique=False, NCHW=True)
        # MBx3xK
        batch_xyz = torch.cat(torch.unbind(batch_xyz, dim=2), dim=0)
        if batch_features is not None:
            # BxCxMxN
            batch_features = torch.unsqueeze(
                batch_features, dim=2).expand(patch_num)
            new_patch_idx = new_patch_idx.unsqueeze(dim=1).expand(
                (-1, batch_features.size(1), -1, -1))  # B, C, M, K
            batch_features = torch.gather(batch_features, 3, new_patch_idx)
            # MBxCxK
            batch_features = torch.cat(
                torch.unbind(batch_features, dim=2), dim=0)

        if gt_xyz is not None and gt_k is not None:
            gt_xyz, _ = operations.group_knn(
                gt_k, gt_xyz, batch_seed_point, unique=False)
            gt_xyz = torch.cat(torch.unbind(gt_xyz, dim=2), dim=0)
        else:
            gt_xyz = None

        return batch_xyz, batch_features, gt_xyz

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
            self.code = self.gen_1d_grid(step_ratio).unsqueeze(0).detach()
        else:
            expansion_ratio = round(sqrt(step_ratio))**2
            self.code = self.gen_grid(expansion_ratio).unsqueeze(0).detach()

        in_channels = 3
        self.layer0 = layers.Conv2d(3, 24, [1, 1],  activation=None)
        self.layer1 = layers.DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 84  # 24+(24+growth_rate*dense_n) = 24+(24+36) = 84
        self.layer2_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer2 = layers.DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 144  # 84+(24+36) = 144
        self.layer3_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer3 = layers.DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 204  # 144+(24+36) = 204
        self.layer4_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer4 = layers.DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 264  # 204+(24+36) = 264
        self.up_layer = torch.nn.Sequential(OrderedDict([
            ("up_layer1", layers.Conv2d(in_channels +
                                        self.code.size(1), 128, 1, activation="relu")),
            ("up_layer2", layers.Conv2d(128, 128, 1, activation="relu")), ]))
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
        distance = torch.mean(
            (points - knn_points) ** 2, dim=1, keepdim=True).detach()
        # mean_P(min_K distance)
        h = torch.mean(torch.mean(distance, dim=-1,
                                  keepdim=True), dim=-2, keepdim=True)
        weight = torch.exp(-distance / (h/2)).detach()
        return distance, weight

    def gen_grid(self, grid_size):
        """
        output [2, grid_size x grid_size]
        """
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        # grid_sizexgrid_size
        x, y = torch.meshgrid(x, x)
        # 2xgrid_sizexgrid_size
        grid = torch.stack([x, y], dim=0).view(
            [2, grid_size*grid_size])  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def gen_1d_grid(self, num_grid_point):
        """
        output [2, num_grid_point]
        """
        grid = torch.linspace(-0.2, 0.2,
                              num_grid_point).view(1, num_grid_point)
        return grid

    def forward(self, xyz, batch_radius, previous_level4=None, gt=None):
        """
        :param
            xyz             Bx3xN input xyz, unnormalized
            batch_radius    Bx1
            previous_level4 tuple of the xyz and feature of the final feature
                            in the previous level (Bx3xM, BxCxM)
        """
        batch_size, _, num_point = xyz.size()
        # normalize
        x, centroid, radius = operations.normalize_point_batch(xyz, NCHW=True)
        x = self.layer0(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        x = torch.cat([self.layer1(x), x], dim=1)
        x = torch.cat([self.layer2(self.layer2_prep(x)), x], dim=1)
        x = torch.cat([self.layer3(self.layer3_prep(x)), x], dim=1)
        x = torch.cat([self.layer4(self.layer4_prep(x)), x], dim=1)
        # interlevel skip connections
        if previous_level4 is not None:
            previous_xyz, previous_feat = previous_level4
            if not self.training and previous_xyz.shape[0] != x.shape[0]:
                # Bx3xM
                previous_xyz = previous_xyz.expand(batch_size, -1, -1)
                # BxCxM
                previous_feat = previous_feat.expand(batch_size, -1, -1)
            # find closest k point in spatial, BxNxK
            knn_points, knn_idx, knn_distance = operations.group_knn(
                self.fm_knn, xyz, previous_xyz, unique=True, NCHW=True)
            # BxCxNxM
            previous_feat = previous_feat.unsqueeze(
                2).expand(-1, -1, num_point, -1, -1)
            # BxCxNxK
            knn_idx = knn_idx.unsqueeze(
                1).expand(-1, previous_feat.size(1), -1, -1)
            # BxCxNxK
            knn_feats = torch.gather(previous_feat, 3, knn_idx)
            # Bx1xNxK
            _, s_average_weight = self.exponential_distance(
                xyz, previous_xyz)
            _, f_average_weight = self.exponential_distance(
                x, previous_feat)
            average_weight = s_average_weight * f_average_weight
            average_weight = average_weight / \
                torch.sum(average_weight+1e-5, dim=-1, keepdim=True)
            # BxCxN
            knn_feats = torch.sum(
                average_weight * knn_feats, dim=-1)

            x = 0.2 * knn_feats + x

        point_features = x
        # BxCxNx1
        x = x.unsqueeze(-1)
        # code 1x1(or2)xr
        _, code_length, ratio = self.code.size()
        # 1x1(or2)x(N*r)
        code = self.code.expand(
            x.size(0), -1, -1).repeat(x.size(0), ratio, x.size(2)*ratio)
        code = batch_radius.unsqueeze(-1) * code
        # BxCxN -> BxCxNxratio
        x = x.unsqueeze(-1).repeat(x.size(0), x.size(1), num_point, ratio)
        x = torch.cat([x, code], dim=1)

        # transform to 3D
        x = self.up_layer(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)

        return x, point_features
