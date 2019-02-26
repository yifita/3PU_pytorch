import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict

import torch
import torch.utils.data as data

from network.upsampler import Net
from model import Model
from network import operations
from utils import pc_utils, pytorch_utils
from misc import logger
from data import H5Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test',
                    help='train or test [default: train]')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--id', default='demo',
                    help="experiment name, prepended to log_dir")
parser.add_argument('--log_dir', default='./model',
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
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size during training')
parser.add_argument('--h5_data', help='h5 file for training')
parser.add_argument('--record_data', help='record file for training')
parser.add_argument('--test_data', help='test data path')
parser.add_argument('--lr_init', type=float, default=0.0005)
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
LR_INIT = FLAGS.lr_init
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
CD_THRESHOLD = FLAGS.cd_threshold

UP_RATIO = FLAGS.up_ratio
TRAIN_H5 = FLAGS.h5_data
TRAIN_RECORD = FLAGS.record_data

TEST_DATA = FLAGS.test_data
PATCH_NUM_RATIO = FLAGS.patch_num_ratio


# build model
net = Net(max_up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
          knn=KNN, growth_rate=GROWTH_RATE, dense_n=DENSE_N, fm_knn=FM_KNN)


def get_stage_progress(step):
    """
    return the stage (an integer from 0) and progress (float 0~1)
    """
    stage = (step + STAGE_STEPS) // (2 * STAGE_STEPS)
    progress = (step + STAGE_STEPS) / (2 * STAGE_STEPS) - stage
    return stage, progress


def train():
    net.to(DEVICE)
    net.train()
    model = Model(net, "train", FLAGS)
    # data loader
    if TRAIN_H5 is not None:
        from data import H5Dataset
        dataset = H5Dataset(
            h5_path=TRAIN_H5,
            num_shape_point=NUM_SHAPE_POINT, num_patch_point=NUM_POINT,
            batch_size=BATCH_SIZE, up_ratio=UP_RATIO, step_ratio=STEP_RATIO)
        dataloader = data.DataLoader(
            dataset, batch_size=1, pin_memory=True, num_workers=4)

    start_epoch = model.step // len(dataloader)
    # whenever progress is changed, we need to update:
    # 1. chamferloss threshold
    # 2. dataset.combined
    # 3. dataset.curr_threshold
    stage, progress = get_stage_progress(model.step)
    start_ratio = STEP_RATIO ** (stage + 1)
    dataset.set_max_ratio(start_ratio)
    if progress > 0.5:
        dataset.set_combined()
        if progress > 0.6:
            model.chamfer_criteria.set_threshold(CD_THRESHOLD)
    else:
        model.chamfer_criteria.unset_threshold()
        dataset.unset_combined()

    dataloader = data.DataLoader(dataset, batch_size=1, pin_memory=True)

    # visualization
    vis_logger = visdom.Visdom(env=FLAGS.id)
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        for i, examples in enumerate(dataloader):
            input_pc, label_pc, ratio = examples
            ratio = ratio.item()
            # 1xBx3xN
            input_pc = input_pc[0].to(DEVICE)
            label_pc = label_pc[0].to(DEVICE)
            model.set_input(input_pc, ratio, label_pc=label_pc)
            # run gradient decent and increment model.step
            model.optimize()
            new_stage, new_progress = get_stage_progress(model.step)
            # advance to the next training stage with an added ratio
            if stage + 1 == new_stage:
                dataset.add_next_ratio()
                dataset.unset_combined()
                model.chamfer_criteria.unset_threshold()
            # advance to the combined stage
            if progress <= 0.5 and new_progress > 0.5:
                dataset.set_combined()
            # chamfer loss set ignore threshold
            if new_progress > 0.6:
                model.chamfer_criteria.set_threshold(CD_THRESHOLD)
            if model.step % 50 == 0:
                output = model.predicted.transpose(2, 1)[0].cpu()
                gt = model.gt.transpose(2, 1)[0].cpu()
                input_pc = input_pc.transpose(2, 1)[0].cpu()
                vis_logger.scatter(input_pc, win="x{}_input".format(ratio),
                                   opts=dict(title="x{}_input".format(ratio),
                                             markersize=2))
                vis_logger.scatter(output, win="x{}_output".format(ratio),
                                   opts=dict(title="x{}_output".format(ratio),
                                             markersize=2))
                vis_logger.scatter(gt, win="x{}_gt".format(ratio),
                                   opts=dict(title="x{}_label".format(ratio),
                                             markersize=2))
                vis_logger.line(
                    np.array([model.error_log["cd_loss_x{}".format(ratio)]]),
                    np.array([model.step]),
                    update="append",
                    win="x{}_loss".format(ratio),
                    opts=dict(title="x{}_loss".format(ratio)))

            stage, progress = new_stage, new_progress

        # end of epoch
        logger.info("epoch %d: " % epoch +
                    ", ".join(["{}={}".format(k, v) for k, v in model.error_log.items()]))
        if epoch % 20 == 0:
            pytorch_utils.save_network(net, MODEL_DIR,
                                       "model", epoch_label=str(epoch),
                                       step=str(model.step))


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
    _, seeds = operations.furthest_point_sample(
        input_pc, num_patches, NCHW=True)
    print("number of patches: %d" % seeds.shape[-1])
    input_list = []
    up_point_list = []

    patches, _, _ = operations.group_knn(
        NUM_POINT, seeds, input_pc, NCHW=True)

    for k in tqdm(range(num_patches)):
        patch = patches[:, :, k, :]
        patch, centroid, radius = operations.normalize_point_batch(
            patch, NCHW=True)
        up_point = net.forward(patch.detach(), ratio=UP_RATIO)
        up_point = up_point * radius + centroid
        input_list.append(patch)
        up_point_list.append(up_point)

    return input_list, up_point_list


def pc_visualization(net, input_pc, patch_num_ratio=3):
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
    _, seeds = operations.furthest_point_sample(
        input_pc, num_patches, NCHW=True)
    print("number of patches: %d" % seeds.shape[-1])
    vis_xyz = defaultdict(list)
    vis_feat = defaultdict(list)
    vis_nnIdx = defaultdict(list)

    patches, _, _ = operations.group_knn(
        NUM_POINT, seeds, input_pc, NCHW=True)

    for k in tqdm(range(num_patches)):
        patch = patches[:, :, k, :]
        net.forward(patch.detach(), ratio=UP_RATIO, phase="vis")
        for k in net.vis:
            if "Idx" in k:
                xyz, nnIdx = net.vis[k]
                vis_nnIdx[k].append(nnIdx)
                vis_xyz[k].append(xyz)
            else:
                xyz, feat = net.vis[k]
                vis_xyz[k].append(xyz)
                vis_feat[k].append(feat)
    return vis_xyz, vis_feat, vis_nnIdx


def vis(result_dir):
    """
    upsample a point cloud
    """
    from sklearn.manifold import TSNE
    from utils.interactive_visualizer import Painter
    # loaded_states = np.load(CKPT).item()
    # net.load_state_dict(loaded_states)
    # pytorch_utils.save_network(net, os.path.dirname(CKPT), "final", "poisson")
    pytorch_utils.load_network(net, CKPT)
    net.to(DEVICE)
    net.eval()
    test_files = glob(TEST_DATA, recursive=True)
    for point_path in test_files:
        folder = os.path.basename(os.path.dirname(point_path))
        out_path = os.path.join(result_dir, folder,
                                point_path.split('/')[-1][:-4] + '.ply')
        data = pc_utils.load(point_path, NUM_SHAPE_POINT)
        data = data[np.newaxis, ...]
        num_shape_point = data.shape[1] * FLAGS.drop_out

        # transpose to NCHW format
        data = torch.from_numpy(data).transpose(2, 1).to(device=DEVICE)

        logger.info(os.path.basename(point_path))
        start = time.time()
        with torch.no_grad():
            # 1x3xN
            xyz_dictlist, feat_dictlist, nnIdx_dictlist = pc_visualization(
                net, data, patch_num_ratio=PATCH_NUM_RATIO)

        for k, v in nnIdx_dictlist.items():
            xyz = xyz_dictlist[k]
            for p in range(1, len(v)):
                # v shape is 1xNxK
                v[p] += v[p - 1].shape[1]
            xyz = torch.cat(xyz, dim=-1)
            xyz = xyz.transpose(2, 1).cpu().numpy()[0, ...]
            nnIdx = torch.cat(v, dim=1)
            nnIdx = nnIdx.cpu().numpy()[0, ...]
            painter = Painter("NN Feature")
            painter.nnIdx = nnIdx
            painter.interactive_3D_plot(xyz, k)


def test(result_dir):
    """
    upsample a point cloud
    """
    pytorch_utils.load_network(net, CKPT)
    net.to(DEVICE)
    net.eval()
    test_files = glob(TEST_DATA, recursive=True)
    for point_path in test_files:
        folder = os.path.basename(os.path.dirname(point_path))
        out_path = os.path.join(result_dir, folder,
                                point_path.split('/')[-1][:-4] + '.ply')
        data = pc_utils.load(point_path, NUM_SHAPE_POINT)
        data = data[np.newaxis, ...]
        num_shape_point = data.shape[1] * FLAGS.drop_out
        if FLAGS.drop_out < 1:
            _, data = operations.furthest_point_sample(
                data, int(num_shape_point))
        # normalize "unnecessarily" to apply noise
        data, centroid, furthest_distance = pc_utils.normalize_point_cloud(
            data)
        is_2D = np.all(data[:, :, 2] == 0)
        if JITTER:
            data = pc_utils.jitter_perturbation_point_cloud(
                data, sigma=FLAGS.jitter_sigma, clip=FLAGS.jitter_max, is_2D=is_2D)

        # transpose to NCHW format
        data = torch.from_numpy(data).transpose(2, 1).to(device=DEVICE)

        logger.info(os.path.basename(point_path))
        start = time.time()
        with torch.no_grad():
            # 1x3xN
            input_pc_list, pred_pc_list = pc_prediction(
                net, data, patch_num_ratio=PATCH_NUM_RATIO)

        # for i, patch_pair in enumerate(zip(input_pc_list, pred_pc_list)):
        #     in_patch, out_patch = patch_pair
        #     pc_utils.save_ply(in_patch.transpose(2, 1).cpu().numpy()[
        #                       0], path[:-4]+'_input_%d.ply' % i)
        #     pc_utils.save_ply(out_patch.transpose(2, 1).cpu().numpy()[
        #                       0], path[:-4]+'_output_%d.ply' % i)
        pred_pc = torch.cat(pred_pc_list, dim=-1)
        input_point = torch.cat(input_pc_list, dim=-1)
        end = time.time()
        print("total time: ", end - start)
        _, pred_pc = operations.furthest_point_sample(
            pred_pc, int(num_shape_point) * UP_RATIO, NCHW=True)
        pred_pc = pred_pc.transpose(2, 1).cpu().numpy()
        pred_pc = (pred_pc * furthest_distance) + centroid
        data = data.transpose(2, 1).cpu().numpy()
        data = (data * furthest_distance) + centroid
        data = data[0, ...]
        pred_pc = pred_pc[0, ...]

        pc_utils.save_ply(data, out_path[:-4] + '_input.ply')
        pc_utils.save_ply(pred_pc, out_path[:-4] + '.ply')


if __name__ == "__main__":
    append_name = []  # type: ignore
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
    elif PHASE == "vis":
        assert(CKPT is not None)
        vis(result_path)
    elif PHASE == "train":
        import visdom
        train()
