import torch
from math import log, sqrt
from collections import OrderedDict

import layers
import operations


class Net(torch.nn.Module):
    """transform high-density patch H to adapt to the large-scale geometry of tlow-density"""

    def __init__(self, dense_n=3, growth_rate=12, knn=16):
        super(Transformer, self).__init__()
        self.dense_n = dense_n
        in_channels = 3
        self.layer0 = layers.Conv2d(3, 24, [1, 1], activation=None)
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

    def gen_grid(self, grid_size):
        """
        output [2, grid_size x grid_size]
        """
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        # grid_sizexgrid_size
        x, y = torch.meshgrid(x, x)
        # 2xgrid_sizexgrid_size
        grid = torch.stack([x, y], dim=0).view(
            [2, grid_size * grid_size])  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def gen_1d_grid(self, num_grid_point):
        """
        output [2, num_grid_point]
        """
        grid = torch.linspace(-0.2, 0.2,
                              num_grid_point).view(1, num_grid_point)
        return grid

    def forward(self, inp, ref, **kwargs):
        """
        :param
            inp             Bx3xN input xyz, unnormalized
            previous_level4 tuple of the xyz and feature of the final feature
                            in the previous level (Bx3xM, BxCxM)
        :return
            xyz             Bx3xNr output xyz, denormalized
        """
        batch_size, _, num_point = inp.size()
        # normalize
        inp_normalized, centroid, radius = operations.normalize_point_batch(
            inp, NCHW=True)

        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis = {}

        x = self.layer0(inp_normalized.unsqueeze(dim=-1)).squeeze(dim=-1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_0"] = x

        y, idx = self.layer1(x)
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_1"] = x
            self.vis["nnIdx_layer_0"] = idx

        y, idx = self.layer2(self.layer2_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_2"] = x
            self.vis["nnIdx_layer_1"] = idx

        y, idx = self.layer3(self.layer3_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_3"] = x
            self.vis["nnIdx_layer_2"] = idx

        y, idx = self.layer4(self.layer4_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_4"] = x
            self.vis["nnIdx_layer_3"] = idx

        # global features
        x = torch.max(x, dim=1, keepdim=True)

        # code 1x1(or2)xr
        _, code_length, ratio = self.code.size()
        # 1x1(or2)x(N*r)
        code = self.code.repeat(x.size(0), 1, num_point)
        code = code.to(device=x.device)
        # BxCxN -> BxCxNxr
        x = x.unsqueeze(-1).repeat(1, 1, 1, ratio)
        # BxCx(N*r)
        x = torch.reshape(
            x, [batch_size, x.size(1), num_point * ratio]).contiguous()
        # Bx(C+1)x(N*r)
        x = torch.cat([x, code], dim=1)

        # transform to 3D coordinates
        # BxCx(N*r)x1
        x = x.unsqueeze(-1)
        x = self.up_layer(x)
        x = self.fc_layer1(x)
        # Bx3x(N*r)
        x = self.fc_layer2(x).squeeze(-1)
        # add residual
        x += torch.reshape(inp_normalized.unsqueeze(3).repeat([1, 1, 1, ratio]), [
            batch_size, 3, num_point * ratio])  # B, N, 4, 3
        # normalize back
        x = (x * radius) + centroid
        return x, point_features, centroid, radius
