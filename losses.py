import torch

import losses

class NmDistanceFunction(torch.autograd.Function):
    """3D point set to 3D point set distance"""
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        B, N, _ = xyz1.size()
        B, M, _ = xyz2.size()
        result = torch.empty(B, N, dtype=xyz1.dtype, device=xyz1.device)
        result_i = torch.empty(B, N, dtype=torch.int32, device=xyz1.device)
        result2 = torch.empty(B, M, dtype=xyz2.dtype, device=xyz2.device)
        result2_i = torch.empty(B, M, dtype=torch.int32, device=xyz2.device)
        result, result_i, result2, result2_i = losses.nmdistance_forward(B, N, xyz1, M, xyz2, result, result_i, result2, result2_i)
        ctx.save_for_backwards = (xyz1, xyz2, result_i, result2_i)
        return result, result_i, result2, result2_i

    @staticmethod
    def backward(ctx, d_dist1, None, d_dist2, None):
        B, N, _ = d_dist1.size()
        B, M, _ = d_dist2.size()
        xyz1, xyz2, idx1, idx2 = ctx.saved_variables
        d_xyz1 = torch.zeros_like(xyz1)
        d_xyz2 = torch.zeros_like(xyz2)
        gradient1, gradient2 = ctx.needs_input_grad
        d_input = losses.nmdistance_backward(B, N, xyz1, M, xyz2, d_dist1, idx1, d_dist2, idx2,
            gradient1, gradient2,
            d_xyz1, d_xyz2)
        if not gradient1:
            return None, d_input[0]
        if not gradient2:
            return d_input[0], None
        else:
            d_input


class ChamferLoss(torch.nn.Module):
    def __init__(self, threshold):
        super(ClassName, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.threshold = threshold

    def forward(self, pred, gt):
        assert(pred.dim() == 3 and gt.dim() == 3), \
            "input for ChamferLoss must be a 3D-tensor, but pred.size() is {} gt.size() is {}".format(pred.size(), gt.size())

        # need transpose
        if pred.size(2) != 3:
            assert(pred.size(1) == 3), "ChamferLoss is implemented for 3D points"
            pred = pred.transpose(2, 1).contiguous()

        if gt.size(2) != 3:
            assert(gt.size(1) == 3), "ChamferLoss is implemented for 3D points"
            gt = gt.transpose(2, 1).contiguous()

        assert(pred.size(2) == 3 and gt.size(2) == 3), "ChamferLoss is implemented for 3D points"
        pred2gt, _, gt2pred, _ = NmDistanceFunction.apply(pred, gt)

        if self.threshold is not None:
            threshold = self.threshold
            forward_threshold = torch.mean(pred2gt, dim=1, keepdim=True) * threshold
            backward_threshold = torch.mean(dists_backward, dim=1, keepdim=True) * threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = torch.where(dists_forward < forward_threshold, dists_forward, torch.zeros_like(dists_forward))
            dists_backward = torch.where(dists_backward < backward_threshold, dists_backward, torch.zeros_like(dists_backward))

        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = torch.mean(dists_forward, dim=1)
        dists_backward = torch.mean(dists_backward, dim=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        # CD_dist_norm = CD_dist/radius
        cd_loss = torch.mean(CD_dist)
        return cd_loss
