import torch
import faiss

import sampling

from faiss_setup import GPU_RES


def normalize_point_cloud(pc, NCHW=False):
    """
    pc [N, P, 3]
    """
    point_axis = 2 if NCHW else 1
    dim_axis = 1 if NCHW else 2
    centroid = torch.mean(pc, dim=point_axis, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.max(
        torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


def extract_xyz_feature_patch(batch_xyz, k, batch_features=None, gt_xyz=None, gt_k=None, patch_num=1):
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
    if patch_num == 1:
        seed_idx = torch.randint(low=0, high=num_point, [batch_size, patch_num], dtype=torch.int32,
            layout=torch.strided, device=batch_xyz.device)
    else:
        assert(batch_size == 1)
        # remove residual,
        _, _, closest_d = group_knn(2, batch_xyz, batch_xyz, unique=False)
        # BxN
        closest_d = closest_d[:, :, 1]
        # BxN, points whose NN is within a threshold Bx1
        mask = closest_d < 5*(torch.mean(closest_d, dim=1, keepdim=True))
        # Bx1xN
        mask = torch.unsqueeze(mask, dim=1).expand_as(batch_xyz)
        # filter (B, P', 3)
        batch_xyz = torch.masked_select(batch_xyz, mask).view(batch_size, 3, num_point)
        patch_num = int(num_point / k * 3)
        batch_xyz_transposed = batch_xyz.transpose(2,1).contiguous()
        idx = furthest_point_sample(batch_xyz_transposed, patch_num)
        batch_seed_point = gather_points(batch_xyz, idx)
        k = torch.min([k, num_point])

    # Bx3xM M=patch_num
    batch_seed_point = gather_points(batch_xyz, seed_idx)
    # Bx3xMxK, BxMxK
    batch_xyz, new_patch_idx, _ = group_knn(k, batch_xyz, batch_seed_point, unique=False, NCHW=True)
    # MBx3xK
    batch_xyz = torch.cat(torch.unbind(batch_xyz, dim=2), dim=0)
    if batch_features is not None:
        # BxCxMxN
        batch_features = torch.unsqueeze(batch_features, dim=2).expand(patch_num)
        new_patch_idx = new_patch_idx.unsqueeze(dim=1).expand((-1, batch_features.size(1), -1, -1))  # B, C, M, K
        batch_features = torch.gather(batch_features, 3, new_patch_idx)
        # MBxCxK
        batch_features = torch.cat(torch.unbind(batch_features, dim=2), dim=0)

    if gt_xyz is not None and gt_k is not None:
        gt_xyz, _ = group_knn(gt_k, gt_xyz, batch_seed_point, unique=False)
        gt_xyz = torch.cat(torch.unbind(gt_xyz, dim=2), dim=0)
    else:
        gt_xyz = None

    return batch_xyz, batch_features, gt_xyz


def search_index_pytorch(database, x, k, D=None, I=None):
    """
    KNN search via Faiss
    :param
        database BxNxC
        x BxMxC
    :return
        D BxMxK
        I BxMxK
    """
    Dptr = database.storage().data_ptr()
    index = faiss.GpuIndexFlatL2(GPU_RES, database.size(-1))  # dimension is 3
    index.add_c(database.size(0), faiss.cast_integer_to_float_ptr(Dptr))

    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        if x.is_cuda:
            D = torch.cuda.FloatTensor(n, k)
        else:
            D = torch.FloatTensor(n, k)
    else:
        assert D.__class__ in (torch.FloatTensor, torch.cuda.FloatTensor)
        assert D.size() == (n, k)
        assert D.is_contiguous()

    if I is None:
        if x.is_cuda:
            I = torch.cuda.LongTensor(n, k)
        else:
            I = torch.LongTensor(n, k)
    else:
        assert I.__class__ in (torch.LongTensor, torch.cuda.LongTensor)
        assert I.size() == (n, k)
        assert I.is_contiguous()
    torch.cuda.synchronize()
    xptr = x.storage().data_ptr()
    Iptr = I.storage().data_ptr()
    Dptr = D.storage().data_ptr()
    index.search_c(n, faiss.cast_integer_to_float_ptr(xptr),
                   k, faiss.cast_integer_to_float_ptr(Dptr),
                   faiss.cast_integer_to_long_ptr(Iptr))
    torch.cuda.synchronize()
    index.reset()
    return D, I


class KNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, query, points, unique=True, NCHW=True):
        """
        :param k: k in KNN
               query: BxMxC
               points: BxNxC
               trans: swap last two axis BxMxC
        :return:
            neighbors_points: BxCxMxK
            index_batch: BxMxK
        """
        batch_size, channels, num_points = points.size()
        if NCHW:
            points_trans = points.transpose(2, 1).contiguous()
            query_trans = query.transpose(2, 1).contiguous()
        else:
            points_trans = points.contiguous()
            query_trans = query.contiguous()

        # selected_gt: BxkxCxM
        # process each batch independently.
        index_batch = []
        distance_batch = []
        for i in range(points.shape[0]):
            # index = self.build_nn_index(points_trans[i])
            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            # _, I_var = search_index_pytorch(index, query_trans[i], k)
            D_var, I_var = search_index_pytorch(points_trans[i], query_trans[i], k)
            GPU_RES.syncDefaultStreamCurrentDevice()
            index_batch.append(I_var)  # M, k
            distance_batch.append(D_var)  # M, k

        index_batch = torch.stack(index_batch, dim=0)  # B, M, K
        distance_batch = torch.stack(distance_batch, dim=0)
        points_expanded = points.unsqueeze(dim=2).expand((-1, -1, query.size(2), -1))  # B, C, M, N
        index_batch_expanded = index_batch.unsqueeze(dim=1).expand((-1, points.size(1), -1, -1))  # B, C, M, k
        neighbor_points = torch.gather(points_expanded, 3, index_batch_expanded)
        index_batch = index_batch.detach()
        return neighbor_points, index_batch, distance_batch

    @staticmethod
    def backward(ctx, d_points, d_index, d_distance):
        return None, None, None


group_knn = KNN.apply


class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        features = features.contiguous()
        idx = idx.contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.empty(B, C, npoint, dtype=features.dtype, device=features.device)
        output = sampling.gather_forward(
            B, C, N, npoint, features, idx, output
        )

        ctx.save_for_backward(idx)
        ctx.C = C
        ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        B, npoint = idx.size()

        grad_features = torch.zeros(B, ctx.C, ctx.N, dtype=grad_out.dtype, device=grad_out.device)
        grad_features = sampling.gather_backward(
            B, ctx.C, ctx.N, npoint, grad_out.contiguous(), idx, grad_features
        )

        return grad_features, None


gather_points = GatherFunction.apply

class FurthestPointSampling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, npoint):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.LongTensor
            (B, npoint) tensor containing the indices
        torch.FloatTensor
            (B, npoint, 3) tensor containing the set
        """
        B, N, _ = xyz.size()

        idx = torch.empty([B, npoint], dtype=torch.int32, device=xyz.device)
        temp = torch.full([B, N], 1e10, dtype=torch.float32, device=xyz.device)

        sampling.furthest_sampling(
            B, N, npoint, xyz, temp, idx
        )
        return idx

    @staticmethod
    def backward(ctx, grad):
        return None, None


furthest_point_sample = FurthestPointSampling.apply

class FurthestPoint(torch.nn.Module):
    """
    Furthest point sampling for Bx3xN points
    param:
        xyz: Bx3XN or BxNx3 tensor
        npoint: number of points
    return:
        idx: Bxnpoint indices
        sampled_xyz: Bx3xnpoint coordinates
    """
    def forward(self, xyz, npoint):
        assert(xyz.dim() == 3), "input for furthest sampling must be a 3D-tensor, but xyz.size() is {}".format(xyz.size())
        # need transpose
        if xyz.size(2) != 3:
            assert(xyz.size(1) == 3), "furthest sampling is implemented for 3D points"
            xyz = xyz.transpose(2, 1).contiguous()

        assert(xyz.size(2) == 3), "furthest sampling is implemented for 3D points"
        idx = furthest_point_sample(xyz, npoint)
        sampled_pc = gather_points(xyz.transpose(2, 1).contiguous(), idx)
        return idx, sampled_pc


if __name__ == '__main__':
    from utils.pc_util import read_ply, save_ply, save_ply_property
    cuda0 = torch.device('cuda:0')
    pc = read_ply("/home/ywang/Documents/points/point-upsampling/3PU/prepare_data/polygonmesh_base/build/data_PPU_output/training/112/angel4_aligned_2.ply")
    pc = pc[:, :3]
    print("{} input points".format(pc.shape[0]))
    save_ply(pc, "./input.ply", colors=None, normals=None)
    pc = torch.from_numpy(pc).requires_grad_().to(cuda0).unsqueeze(0)
    pc = pc.transpose(2, 1)

    # test furthest point
    furthest_point = FurthestPoint()
    idx, sampled_pc = furthest_point(pc, 1250)
    output = sampled_pc.transpose(2, 1).cpu().squeeze()
    save_ply(output.detach(), "./output.ply", colors=None, normals=None)

    # test KNN
    knn_points, _, _ = group_knn(10, sampled_pc, pc)  # B, C, M, K
    labels = torch.arange(0, knn_points.size(2)).unsqueeze_(0).unsqueeze_(0).unsqueeze_(-1)  # 1, 1, M, 1
    labels = labels.expand(knn_points.size(0), -1, -1, knn_points.size(3))  # B, 1, M, K
    # B, C, P
    labels = torch.cat(torch.unbind(labels, dim=-1), dim=-1).squeeze().detach().cpu().numpy()
    knn_points = torch.cat(torch.unbind(knn_points, dim=-1), dim=-1).transpose(2, 1).squeeze(0).detach().cpu().numpy()
    save_ply_property(knn_points, labels, "./knn_output.ply", cmap_name='jet')

    from torch.autograd import gradcheck

    test = gradcheck(gather_points, [pc.to(dtype=torch.float64), idx], eps=1e-6, atol=1e-4)
    print(test)
