import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from glob import glob

import torch

from net import Net
from model import Model
from utils import pc_utils, pytorch_utils
from misc import logger
import operations

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test',
                    help='train or test [default: train]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--id', default='demo',
                    help="experiment name, prepended to log_dir")
parser.add_argument('--log_dir', default='../model',
                    help='Log dir [default: log]')
parser.add_argument('--model', default='model_microscope', help='model name')
parser.add_argument('--root_dir', default='../',
                    help='project root, data and h5_data diretories')
parser.add_argument('--result_dir', help='result directory')
parser.add_argument('--ckpt', help='model to restore from')
parser.add_argument('--num_point', type=int,
                    help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--num_shape_point', type=int,
                    help="Number of points per shape")
parser.add_argument('--up_ratio', type=int, default=16,
                    help='Upsampling Ratio [default: 2]')
parser.add_argument('--max_epoch', type=int, default=160,
                    help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=28,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--h5_data', help='h5 file for training')
parser.add_argument('--record_data', help='record file for training')
parser.add_argument('--test_data', help='test data path')
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--restore_epoch', type=int)
parser.add_argument('--stage_steps', type=int, default=15000,
                    help="number of updates per curriculums stage")
parser.add_argument('--step_ratio', type=int, default=2,
                    help="upscale ratio per step")
parser.add_argument('--patch_num_ratio', type=float, default=3)
parser.add_argument('--jitter', action="store_true",
                    help="jitter augmentation")
parser.add_argument('--jitter_sigma', type=float,
                    default=0.0025, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float,
                    default=0.005, help="jitter augmentation")
parser.add_argument('--drop_out', type=float, default=1.0,
                    help="drop_out ratio. default 1.0 (no drop out) ")
parser.add_argument('--knn', type=int, default=32,
                    help="neighbood size for edge conv")
parser.add_argument('--dense_n', type=int, default=3,
                    help="number of dense layers")
parser.add_argument('--block_n', type=int, default=3,
                    help="number of dense blocks")
parser.add_argument('--fm_knn', type=int, default=5,
                    help="number of neighboring points for feature matching")

parser.add_argument('--growth_rate', type=int, default=12,
                    help='dense block growth rate')
parser.add_argument('--cd_threshold', default=2.0,
                    type=float, help="threshold for cd")
parser.add_argument('--fidelity_weight', default=50.0,
                    type=float, help="chamfer loss weight")

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
DEVICE = torch.device('cuda', FLAGS.gpu)
ROOT_DIR = FLAGS.root_dir
MODEL_DIR = os.path.join(FLAGS.log_dir, FLAGS.id)
CKPT = FLAGS.ckpt

NUM_SHAPE_POINT = FLAGS.num_shape_point
NUM_POINT = FLAGS.num_point
assert(NUM_SHAPE_POINT is not None or NUM_POINT is not None)
NUM_POINT = NUM_POINT or int(NUM_SHAPE_POINT * FLAGS.drop_out)

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
JITTER = FLAGS.jitter
JITTER_MAX = FLAGS.jitter_max
JITTER_SIGMA = FLAGS.jitter_sigma
STAGE_STEPS = FLAGS.stage_steps

STEP_RATIO = FLAGS.step_ratio
RESTORE_EPOCH = FLAGS.restore_epoch
FM_KNN = FLAGS.fm_knn
KNN = FLAGS.knn
GROWTH_RATE = FLAGS.growth_rate
DENSE_N = FLAGS.dense_n

UP_RATIO = FLAGS.up_ratio
TRAIN_H5 = FLAGS.h5_data
TRAIN_RECORD = FLAGS.record_data

TEST_DATA = FLAGS.test_data
PATCH_NUM_RATIO = FLAGS.patch_num_ratio


# build model
net = Net(max_up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
          knn=KNN, growth_rate=GROWTH_RATE, dense_n=DENSE_N, fm_knn=FM_KNN)


def pc_prediction(net, input_pc, patch_num_ratio=3):
    """
    upsample patches of a point cloud
    :param
        input_pc        1x3xN
        patch_num_ratio int, impacts number of patches and overlapping
    :return
        input_list      list of [3xM]
        up_point_list   list of [3xMr]
    """
    # divide to patches
    num_patches = int(input_pc.shape[2] / NUM_POINT * patch_num_ratio)
    # FPS sampling
    start = time.time()
    _, seeds = operations.furthest_point_sample(input_pc, num_patches, NCHW=True)
    print("number of patches: %d" % seeds.shape[0])
    input_list = []
    up_point_list = []

    patches, _, _ = operations.group_knn(
        NUM_POINT, seeds, input_pc, NCHW=True)

    for k in tqdm(range(num_patches)):
        patch = patches[:, :, k, :]
        up_point = net.forward(patch.detach(), ratio=UP_RATIO)
        input_list.append(patch)
        up_point_list.append(up_point)

    # up_point = torch.cat(up_point_list, dim=-1)
    # input_point = torch.cat(input_list, dim=-1)

    return input_list, up_point_list


def test(result_dir):
    """
    upsample a point cloud
    """
    # loaded_states = np.load(CKPT).item()
    # net.load_state_dict(loaded_states)
    # pytorch_utils.save_network(net, os.path.dirname(CKPT), "final", "poisson")
    pytorch_utils.load_network(net, CKPT)
    net.to(DEVICE)
    net.eval()
    test_files = glob(TEST_DATA, recursive=True)
    for point_path in test_files:
        folder = os.path.basename(os.path.dirname(point_path))
        path = os.path.join(result_dir, folder,
                            point_path.split('/')[-1][:-4]+'.ply')
        data = pc_utils.load(point_path, NUM_SHAPE_POINT)
        data = data[np.newaxis, ...]
        num_shape_point = data.shape[1] * FLAGS.drop_out
        # normalize "unnecessarily" to apply noise
        data, centroid, furthest_distance = pc_utils.normalize_point_cloud(
            data)
        is_2D = np.all(data[:, :, 2] == 0)
        if FLAGS.drop_out < 1:
            _, data = operations.furthest_point_sample(data, int(num_shape_point))
        if JITTER:
            data = pc_utils.jitter_perturbation_point_cloud(
                data, sigma=FLAGS.jitter_sigma, clip=FLAGS.jitter_max, is_2D=is_2D)

        # transpose to NCHW format
        data = torch.from_numpy(data).transpose(2, 1).to(device=DEVICE)
        # get the edge information
        logger.info(os.path.basename(point_path))
        start = time.time()
        with torch.no_grad():
            # 1x3xN
            input_pc_list, pred_pc_list = pc_prediction(
                net, data, patch_num_ratio=PATCH_NUM_RATIO)

        for i, patch_pair in enumerate(zip(input_pc_list, pred_pc_list)):
            in_patch, out_patch = patch_pair
            pc_utils.save_ply(in_patch.transpose(2,1).cpu().numpy()[0], path[:-4]+'_input_%d.ply' % i)
            pc_utils.save_ply(out_patch.transpose(2,1).cpu().numpy()[0], path[:-4]+'_output_%d.ply' % i)
        pred_pc = torch.cat(pred_pc_list, dim=-1)
        input_point = torch.cat(input_pc_list, dim=-1)
        end = time.time()
        print("total time: ", end-start)
        _, pred_pc = operations.furthest_point_sample(
            pred_pc, int(num_shape_point)*UP_RATIO, NCHW=True)
        pred_pc = pred_pc.transpose(2, 1).cpu().numpy()
        pred_pc = (pred_pc * furthest_distance) + centroid
        data = data.transpose(2, 1).cpu().numpy()
        data = (data * furthest_distance) + centroid
        data = data[0,...]
        pred_pc = pred_pc[0,...]

        # data = input_pc.transpose(2, 1).cpu().numpy()
        # data = (data * furthest_distance) + centroid
        # data = data[0,...]
        pc_utils.save_ply(data, path[:-4]+'_input.ply')
        pc_utils.save_ply(pred_pc, path[:-4]+'.ply')


if __name__ == "__main__":
    append_name = []
    if NUM_POINT is None:
        append_name += ["pWhole"]
    else:
        append_name += ["p%d" % NUM_POINT]
    if NUM_SHAPE_POINT is None:
        append_name += ["sWhole"]
    else:
        append_name += ["s%d" % NUM_SHAPE_POINT]

    if JITTER:
        append_name += ["s{}".format("{:.4f}".format(
            FLAGS.jitter_sigma).replace(".", ""))]
    else:
        append_name += ["clean"]

    if FLAGS.drop_out < 1:
        append_name += ["d{}".format(
            "{:.2f}".format(FLAGS.drop_out).replace(".", ""))]

    result_path = FLAGS.result_dir or os.path.join(
        MODEL_DIR, 'result', 'x%d' % (UP_RATIO), "_".join(append_name))
    if PHASE == "test":
        assert(CKPT is not None)
        test(result_path)
