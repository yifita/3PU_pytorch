import torch
import h5py
import re
import os
from math import log
import numpy as np
import copy

from utils import multiproc_dataloader as multiproc
from utils import pc_utils
from operations import group_knn


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_shape_point, num_patch_point,
                 phase="train",
                 up_ratio=16, step_ratio=2,
                 jitter=False, jitter_max=0.01, jitter_sigma=0.005,
                 batch_size=16, drop_out=1.0):
        super(H5Dataset, self).__init__()
        np.random.seed(0)
        self.is_2D = False
        self.batch_size = batch_size
        self.num_patch_point = num_patch_point
        self.num_shape_point = num_shape_point
        self.jitter = jitter
        self.jitter_max = jitter_max
        self.jitter_sigma = jitter_sigma
        self.drop_out = drop_out
        self.step_ratio = step_ratio
        self.input_array, self.label_array = self.load_patch_data(
            h5_path, up_ratio, step_ratio, num_shape_point)

    def load_patch_data(self, h5_path, up_ratio, step_ratio, num_point):
        """
        read point inputs and ground truth from h5 file into memory.
        h5 file name is train_{dataset1}_{dataset2}.hdf5
        dataset names are composed of {label}_{num_point}
        :param
            h5_path: string to h5_path
            up_ratio: integer upscaling ratio
            step_ratio: integer upscaling ratio of each step
            num_point: number of points in the input shape
        :return
            data: BxNx3 float
            label: dict label["x{ratio}"] Bx(ratio*N)x3
        """
        h5_filepath = os.path.join(h5_path)
        num_points = re.findall("\d+", os.path.basename(h5_filepath)[:-5])
        num_points = list(map(int, num_points))
        num_points.sort()
        num_points = np.asarray(num_points)
        num_in_point = num_points[np.searchsorted(num_points, num_point)]

        f = h5py.File(h5_filepath, "r")
        tag = re.findall("_([A-Za-z]+)_", os.path.basename(h5_filepath))[-1]

        data = f[tag+"_%d" % num_in_point][:, :, 0:3]
        logger.info("input point_num %d" % data.shape[1])

        centroid = np.mean(data[:, :, 0:3], axis=1, keepdims=True)
        data[:, :, 0:3] = data[:, :, 0:3] - centroid
        furthest_distance = np.amax(
            np.sqrt(np.sum(data[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        data[:, :, 0:3] = data[:, :, 0:3] / \
            np.expand_dims(furthest_distance, axis=-1)
        label = {}

        self.scale = []
        for x in range(1, int(log(up_ratio, step_ratio)+1)):
            r = step_ratio**x
            self.scale.append(r)
            closest_larger_equal = num_points[np.searchsorted(
                num_points, num_in_point*r)]
            label["x%d" % r] = f[tag+"_%d" % closest_larger_equal][:, :, :3]
            label["x%d" % r][:, :, 0:3] = label["x%d" %
                                                r][:, :, 0:3] - centroid
            label["x%d" % r][:, :, 0:3] = label["x%d" % r][:, :, 0:3] / \
                np.expand_dims(furthest_distance, axis=-1)
            logger.info("gt (ratio %d), point_num %d " %
                        (r, label["x%d" % r].shape[1]))

        f.close()

        if np.all(data[:, :, 2] == 0):
            self.is_2D = True
            logger.info("2D dataset")

        logger.info(("total %d samples" % (data.shape[0])))
        return data, label

    def shape_to_patch(self, input_pc, label_pc):
        """
        sample random patch from the input shapes
        :param
            input_pc: (1, N, 3)
            label_pc: (1, r*N, 3)
        :param
            input_patch: (B, M, 3)
            label_patch: (B, r*M, 3)
        """
        rnd_pts = np.random.randint(0, input_pc.shape[1], [self.batch_size])
        rnd_pts = input_pc[:, rnd_pts, :]  # [batch_size, 1, 3]
        # [1, B, rK, 3]
        label_patches = group_knn(
            self.num_patch_point*ratio, label_pc, rnd_pts, NCHW=False)[0][0]
        # [1, B, K, 3]
        input_patches = group_knn(
            self.num_patch_point, input_pc, rnd_pts, NCHW=False)[0][0]

        return input_patches, label_patches

    def augment(self, input_patches, label_patches):
        """
        augment data with noise, rotation, scaling
        """
        if self.jitter:
            input_pc = pc_utils.jitter_perturbation_point_cloud(
                input_pc, sigma=self.jitter_sigma, clip=self.jitter_max)

        input_patches, label_patches = pc_utils.rotate_point_cloud_and_gt(
            input_patches, label_patches)

        # normalize using the same mean and radius
        label_patches, centroid, furthest_distance = pc_utils.normalize_point_cloud(
            label_patches)
        input_patches = (input_patches - centroid) / furthest_distance

        input_patches, label_patches, scales = pc_utils.random_scale_point_cloud_and_gt(
            input_patches, label_patches,
            scale_low=0.8, scale_high=1.2)

        # randomly discard input
        if self.drop_out < 1:
            num_point = input_patches.shape[1].value * self.drop_out
            point_idx = np.random.shuffle(np.arange(self.num_patch_point))[
                :num_point]
            input_patches = input_patches[:, point_idx, :]

        return input_patches, label_patches, scales

    def __getitem__(self, index):
        self.get(index, None)

    def get(self, index, ratio=None):
        if ratio is None:
            ratio = np.random.choice(self.scale)

        input_patches, label_patches = self.shape_to_patch(
            self.input_array[index:index+1, ...], self.label_array["x%d" % ratio][index:index+1, ...])

        # augment data
        if self.phase == "train":
            input_patches, label_patches, scales = self.augment(
                input_patches, label_patches)
        else:
            # normalize using the same mean and radius
            label_patches, centroid, furthest_distance = pc_utils.normalize_point_cloud(
                label_patches)
            input_patches = (input_patches - centroid) / furthest_distance
            scales = np.ones([B, 1], dtype=np.float32)

        input_patches = torch.from_numpy(input_patches).transpose(2, 1)
        label_patches = torch.from_numpy(label_patches).transpose(2, 1)
        scales = torch.from_numpy(scales)

        return input_patches, label_patches, scales


class DataLoader(multiproc.MyDataLoader):
    """Hacky way to progressively load scales"""

    def __init__(self, dataset, batch_size, scale=None):
        self.dataset = dataset
        super(DataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.phase == Phase.TRAIN),
            num_workers=16,
            random_vars=copy.deepcopy(
                dataset.scale) if self.phase == "train" else None,
            sampler=None)


if __name__ == "__main__":
    pass
