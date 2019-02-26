import torch
from math import log, sqrt
from collections import OrderedDict

from . import layers
from . import operations


class Net(torch.nn.Module):
    """3PU inter-level plus skip connection and dense layers"""

    def __init__(self, max_up_ratio=16, step_ratio=2, knn=16, growth_rate=12,
                 dense_n=3, max_num_point=312, fm_knn=3, **kwargs):
        super(Net, self).__init__()
        self.max_up_ratio = max_up_ratio
        self.step_ratio = step_ratio
        self.knn = knn
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.fm_knn = fm_knn
        self.num_levels = int(log(max_up_ratio, step_ratio))
        self.levels = torch.nn.ModuleDict()
        self.max_num_point = max_num_point
        for l in range(1, self.num_levels + 1):
            self.levels['level_%d' % l] = Level(
                dense_n=dense_n, growth_rate=growth_rate, knn=knn, step_ratio=step_ratio)
        if self.training:
            for m in self.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)
                elif isinstance(m, (torch.nn.InstanceNorm1d,
                                    torch.nn.InstanceNorm2d,
                                    torch.nn.BatchNorm1d,
                                    torch.nn.BatchNorm2d)):
                    torch.nn.init.zeros_(m.bias)
                    torch.nn.init.ones_(m.weight)

    def extract_xyz_feature_patch(self, batch_xyz, k, gt_xyz=None, gt_k=None):
        """
        extract patches via KNN from input point and ground truth patches
        :param
            batch_xyz: Bx3xN
            k:         patch size
            gt_xyz:    Bx3xM
            gt_k:      size of ground truth patch
        :return
            batch_xyz  MBx3xK
            gt_xyz:    M'Bx3xK  ground truth patch
        """
        batch_size, _, num_point = batch_xyz.size()
        if self.training:
            # randomly choose a point as the seed for KNN extraction
            patch_num = 1
            seed_idx = torch.randint(low=0, high=num_point, size=[batch_size, patch_num], dtype=torch.int32,
                                     layout=torch.strided, device=batch_xyz.device)
            # Bx3xM M=1
            batch_seed_point = operations.gather_points(batch_xyz, seed_idx)
        else:
            # sample uniform KNN seeds
            assert(batch_size == 1)
            # distance to the closest neighbor, if too large, consider as an outlier
            _, _, closest_d = operations.group_knn(
                2, batch_xyz, batch_xyz, unique=False, NCHW=True)
            # BxN closest distance
            closest_d = closest_d[:, :, 1]
            # BxN, points whose NN is within a threshold (Bx1)
            mask = closest_d < (5 * (torch.mean(closest_d, dim=1, keepdim=True)))
            # BxCxN
            mask = torch.unsqueeze(mask, dim=1).expand_as(batch_xyz)
            # filter (B, 3，N')
            batch_xyz = torch.masked_select(
                batch_xyz, mask).view(1, 3, -1)

            num_point = batch_xyz.size(2)
            patch_num = int(num_point / k * 5)
            # Bx3xpatch_num
            _, batch_seed_point = operations.furthest_point_sample(
                batch_xyz, patch_num)
            k = min(k, num_point)

        # Bx3xMxK, BxMxK
        batch_xyz, new_patch_idx, _ = operations.group_knn(
            k, batch_seed_point, batch_xyz, unique=False, NCHW=True)
        # MBx3xK
        batch_xyz = torch.cat(torch.unbind(batch_xyz, dim=2), dim=0)
        # if batch_features is not None:
        #     # BxCxMxN
        #     batch_features = torch.unsqueeze(
        #         batch_features, dim=2).expand(patch_num)
        #     new_patch_idx = new_patch_idx.unsqueeze(dim=1).expand(
        #         (-1, batch_features.size(1), -1, -1))  # B, C, M, K
        #     batch_features = torch.gather(batch_features, 3, new_patch_idx)
        #     # MBxCxK
        #     batch_features = torch.cat(
        #         torch.unbind(batch_features, dim=2), dim=0)

        if gt_xyz is not None and gt_k is not None:
            gt_xyz, _, _ = operations.group_knn(
                gt_k, batch_seed_point, gt_xyz, unique=False)
            gt_xyz = torch.cat(torch.unbind(gt_xyz, dim=2), dim=0)
        else:
            gt_xyz = None

        return batch_xyz, gt_xyz

    def forward(self, xyz, ratio=None, gt=None, **kwargs):
        """
        :param
            xyz     Bx3xN
            ratio   upscaling factor (integer)
            gt      Bx3x(max_up_ratio*N)
        :return
            xyz     Bx3x(ratio*N)
            (during training:)
            gt      Bx3x(ratio*N)
        """
        ratio = ratio or self.max_up_ratio
        if self.training:
            assert(gt is not None)

        batch_size, _, num_point = xyz.size()
        num_levels = int(log(ratio, self.step_ratio))
        max_num_point = min(num_point, self.max_num_point)

        for l in range(1, num_levels + 1):
            curr_ratio = self.step_ratio ** l
            if l > 1:
                # extract input to patches
                if xyz.size(-1) > max_num_point:
                    gt_k = max_num_point * ratio // curr_ratio * self.step_ratio
                    # patches of xyz and feature, but unnormalized
                    patch_xyz, gt = self.extract_xyz_feature_patch(
                        xyz, max_num_point, gt_k=gt_k, gt_xyz=gt)
                else:
                    patch_xyz = xyz

                patch_xyz_normalized, centroid, radius = operations.normalize_point_batch(
                    patch_xyz, NCHW=True)
                # Bx3x(N*r) and BxCx(N*r)
                xyz, features = self.levels['level_%d' % l](
                    patch_xyz, patch_xyz_normalized, previous_level4=(old_xyz, old_features),
                    **kwargs)
                xyz = xyz * radius + centroid
                # cache input xyz for feature propagation
                old_xyz = patch_xyz
                old_features = features
                # merge patches in testing
                if not self.training and (patch_xyz.shape[0] != batch_size):
                    xyz = torch.cat(
                        torch.split(xyz, batch_size, dim=0), dim=2)
                    old_xyz = torch.cat(
                        torch.split(old_xyz, batch_size, dim=0), dim=2)
                    old_features = torch.cat(
                        torch.split(old_features, batch_size, dim=0), dim=2)
                    num_output_point = num_point * curr_ratio
                    # resample to get sparser points idx [B, P, 1]
                    _, xyz = operations.furthest_point_sample(
                        xyz, num_output_point)
            else:
                # Bx3x(N*r) and BxCx(N*r)
                old_xyz = xyz
                xyz, features = self.levels['level_%d' % l](
                    xyz, xyz, previous_level4=None, **kwargs)
                old_features = features

            # for visualization
            if "phase" in kwargs and kwargs["phase"] == "vis":
                self.vis = {}
                for k, v in self.levels["level_%d" % l].vis.items():
                    if "Idx" in k:
                        # v has (B, N, k) change to (1, B*N, k)
                        # add index offset (B, 1, 1)
                        offset = torch.arange(0, v.shape[0],
                                              device=v.device).reshape(-1, 1, 1)
                        offset *= v.shape[1]
                        v += offset
                        v = torch.cat(torch.split(v, batch_size, dim=0), dim=1)
                        self.vis["level_{}.{}".format(l, k)] = (old_xyz, v)
                    else:
                        v = torch.cat(
                            torch.split(v, batch_size, dim=0), dim=2)
                        self.vis["level_{}.{}".format(l, k)] = (old_xyz, v)
                self.vis["level_%d" % l] = (old_xyz, old_features)

        if self.training:
            return xyz, gt
        else:
            return xyz


class Level(torch.nn.Module):
    """3PU per-level network"""

    def __init__(self, dense_n=3, growth_rate=12, knn=16, fm_knn=5, step_ratio=2):
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
            ("up_layer1", layers.Conv2d(in_channels
                                        + self.code.size(1), 128, 1, activation="relu")),
            ("up_layer2", layers.Conv2d(128, 128, 1, activation="relu")), ]))
        self.fc_layer1 = layers.Conv2d(128, 64, 1, activation="relu")
        self.fc_layer2 = layers.Conv2d(64, 3, 1, activation=None)

    def exponential_distance(self, points, knnIdx_points):
        """
        compute knn point distance and interpolation weight for interlevel skip connections
        :param
            points      BxCxN
            knnIdx_points  BxCxNxK
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        if points.dim() == 3:
            points = points.unsqueeze(dim=-1)
        distance = torch.sum(
            (points - knnIdx_points) ** 2, dim=1, keepdim=True).detach()
        # mean_P(min_K distance)
        h = torch.mean(torch.min(distance, dim=-1,
                                 keepdim=True)[0], dim=-2, keepdim=True)
        weight = torch.exp(-distance / (h / 2)).detach()
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
            [2, grid_size * grid_size])  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def gen_1d_grid(self, num_grid_point):
        """
        output [2, num_grid_point]
        """
        grid = torch.linspace(-0.2, 0.2,
                              num_grid_point).view(1, num_grid_point)
        return grid

    def forward(self, xyz, xyz_normalized, previous_level4=None, **kwargs):
        """
        :param
            xyz             Bx3xN input xyz, unnormalized
            xyz_normalized  Bx3xN input xyz, normalized
            previous_level4 tuple of the xyz and feature of the final feature
                            in the previous level (Bx3xM, BxCxM)
        :return
            xyz             Bx3xNr output xyz, normalized
            l4_features     BxCxN feature of the input points
        """
        batch_size, _, num_point = xyz_normalized.size()

        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis = {}

        x = self.layer0(xyz_normalized.unsqueeze(dim=-1)).squeeze(dim=-1)
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

        # interlevel skip connections
        if previous_level4 is not None and self.fm_knn > 0:
            previous_xyz, previous_feat = previous_level4
            if not self.training and previous_xyz.shape[0] != x.shape[0]:
                # Bx3xM
                previous_xyz = previous_xyz.expand(batch_size, -1, -1)
                # BxCxM
                previous_feat = previous_feat.expand(batch_size, -1, -1)
            # find closest k point in spatial, BxNxK
            knnIdx_points, knnIdx_idx, _ = operations.group_knn(
                self.fm_knn, xyz, previous_xyz, unique=True, NCHW=True)
            # BxCxNxM
            previous_feat = previous_feat.unsqueeze(
                2).expand(-1, -1, num_point, -1)
            # BxCxNxK
            knnIdx_idx = knnIdx_idx.unsqueeze(
                1).expand(-1, previous_feat.size(1), -1, -1)
            # BxCxNxK
            knnIdx_feats = torch.gather(previous_feat, 3, knnIdx_idx)
            # Bx1xNxK
            _, s_average_weight = self.exponential_distance(
                xyz, knnIdx_points)
            _, f_average_weight = self.exponential_distance(
                x, knnIdx_feats)
            average_weight = s_average_weight * f_average_weight
            average_weight = average_weight / \
                torch.sum(average_weight + 1e-5, dim=-1, keepdim=True)
            # BxCxN
            knnIdx_feats = torch.sum(
                average_weight * knnIdx_feats, dim=-1)

            x = 0.2 * knnIdx_feats + x

        point_features = x
        # code 1x1(or2)xr
        _, code_length, ratio = self.code.size()
        # 1x1(or2)x(N*r)
        code = self.code.repeat(x.size(0), 1, num_point)
        code = code.to(device=x.device)
        # BxCxN -> BxCxNxr
        x = x.unsqueeze(-1).expand(-1, -1, -1, ratio)
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
        x += torch.reshape(xyz_normalized.unsqueeze(3).repeat([1, 1, 1, ratio]), [
            batch_size, 3, num_point * ratio])  # B, N, 4, 3

        return x, point_features


class AdaptiveLevel(Level):
    """
    Upsampling unit with undeterministic target point number
    """

    def __init__(self, dense_n=3, growth_rate=12, knn=16, fm_knn=5):
        super(Level, self).__init__()
        self.dense_n = dense_n
        self.fm_knn = fm_knn
        in_channels = 3
        self.layer0 = layers.Conv2d(3, 24, [1, 1], activation=None)
        self.layer1 = layers.DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 84  # 24+(24+growth_rate*dense_n) = 24+(24+36) = 84
        self.layer2_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer2 = layers.SampledDenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 144  # 84+(24+36) = 144
        self.layer3_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer3 = layers.SampledDenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 204  # 144+(24+36) = 204
        self.layer4_prep = layers.Conv1d(in_channels, 24, 1, activation="relu")
        self.layer4 = layers.SampledDenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 264  # 204+(24+36) = 264
        self.up_layer = torch.nn.Sequential(OrderedDict([
            ("up_layer1", layers.Conv2d(in_channels + 2, 128, 1, activation="relu")),
            ("up_layer2", layers.Conv2d(128, 128, 1, activation="relu")), ]))
        self.fc_layer1 = layers.Conv2d(128, 64, 1, activation="relu")
        self.fc_layer2 = layers.Conv2d(64, 3, 1, activation=None)

    def exponential_distance(self, points, knnIdx_points):
        """
        compute knn point distance and interpolation weight for interlevel skip connections
        :param
            points      BxCxN
            knnIdx_points  BxCxNxK
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        if points.dim() == 3:
            points = points.unsqueeze(dim=-1)
        distance = torch.sum(
            (points - knnIdx_points) ** 2, dim=1, keepdim=True).detach()
        # mean_P(min_K distance)
        h = torch.mean(torch.min(distance, dim=-1,
                                 keepdim=True)[0], dim=-2, keepdim=True) + 1e-5
        weight = torch.exp(-distance / (h / 2)).detach()
        return distance, weight

    def gen_grid(self, grid_size):
        """
        output [2, grid_size x grid_size]
        """
        x = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
        # grid_sizexgrid_size
        x, y = torch.meshgrid(x, x)
        # 2xgrid_sizexgrid_size
        grid = torch.stack([x, y], dim=0).view(
            [2, grid_size * grid_size])  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def interpolate(self, previous_xyz, xyz, previous_feat):
        # interpolate (B,C,N) to (B,C,N')
        # find closest k point in spatial, BxNxK
        batch, _, num_point = xyz.size()
        knnIdx_points, knnIdx_idx, _ = operations.group_knn(
            self.fm_knn, xyz, previous_xyz, unique=True, NCHW=True)
        # BxCxNxM
        previous_feat = previous_feat.unsqueeze(
            2).repeat(1, 1, num_point, 1)
        # BxCxNxK
        knnIdx_idx = knnIdx_idx.unsqueeze(
            1).repeat(1, previous_feat.size(1), 1, 1)
        # BxCxNxK
        knnIdx_feats = torch.gather(previous_feat, 3, knnIdx_idx)
        # Bx1xNxK
        _, s_average_weight = self.exponential_distance(
            xyz, knnIdx_points)

        average_weight = s_average_weight
        average_weight = average_weight / \
            torch.sum(average_weight + 1e-5, dim=-1, keepdim=True)
        # BxCxN
        knnIdx_feats = torch.sum(
            average_weight * knnIdx_feats, dim=-1)
        return knnIdx_feats

    def forward(self, xyz, target_n_point):
        # 2xn
        code = self.gen_grid(round(sqrt(target_n_point)))
        code = code.expand(xyz.size(0), -1, -1)
        batch_size, _, num_point = xyz.size()

        # normalize
        xyz_normalized, centroid, radius = operations.normalize_point_batch(
            xyz, NCHW=True)

        x = self.layer0(xyz_normalized.unsqueeze(dim=-1)).squeeze(dim=-1)

        y, idx = self.layer1(x)
        x = torch.cat([y, x], dim=1)
        # (B,C',nsample), (B,3, nsample), (B,nsample)
        y, _sampled_xyz, sampled_idx = self.layer2(self.layer2_prep(x), 48, xyz_normalized)
        x = torch.cat([y, self.interpolate(xyz_normalized, _sampled_xyz, x)], dim=1)
        sampled_xyz = _sampled_xyz

        y, _sampled_xyz, sampled_idx = self.layer3(self.layer3_prep(x), 16, sampled_xyz)
        x = torch.cat([y, self.interpolate(sampled_xyz, _sampled_xyz, x)], dim=1)
        sampled_xyz = _sampled_xyz

        y, _sampled_xyz, sampled_idx = self.layer4(self.layer4_prep(x), 1, sampled_xyz)
        x = torch.cat([y, self.interpolate(sampled_xyz, _sampled_xyz, x)], dim=1)
        sampled_xyz = _sampled_xyz

        global_features = x

        code = code.to(device=x.device)
        # BxCx(N*r)
        x = x.expand(-1, -1, code.size(-1))
        # Bx(C+1)x(N*r)
        x = torch.cat([x, code], dim=1)

        # transform to 3D coordinates
        # BxCx(N*r)x1
        x = x.unsqueeze(-1)
        x = self.up_layer(x)
        x = self.fc_layer1(x)
        # Bx3x(N*r)
        x = self.fc_layer2(x).squeeze(-1)

        # normalize back
        x = (x * radius.detach()) + centroid.detach()
        return x, global_features
