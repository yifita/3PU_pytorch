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
        losses.nmdistance_forward(xyz1, xyz2, result, result2, result_i, result2_i)
        ctx.save_for_backward(xyz1, xyz2, result_i, result2_i)
        ctx.mark_non_differentiable(result_i, result2_i)
        return result, result_i, result2, result2_i

    @staticmethod
    def backward(ctx, graddist1, gradNone1, graddist2, gradNone2):
        B, N = d_dist1.size()
        B, M = d_dist2.size()
        xyz1, xyz2, idx1, idx2 = ctx.saved_variables
        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2

nndistance = NmDistanceFunction.apply


class ChamferLoss(torch.nn.Module):
    """
    chamfer loss. bidirectional nearest neighbor distance of two point sets.
    """

    def __init__(self, threshold=None, forward_weight=1.0):
        super(ChamferLoss, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.__threshold = threshold
        self.forward_weight = forward_weight

    def set_threshold(self, value):
        self.__threshold = value

    def unset_threshold(self):
        self.__threshold = None

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

        assert(pred.size(2) == 3 and gt.size(2) ==
               3), "ChamferLoss is implemented for 3D points"
        pred2gt, _, gt2pred, _ = NmDistanceFunction.apply(pred, gt)

        if self.__threshold is not None:
            threshold = self.__threshold
            forward_threshold = torch.mean(
                pred2gt, dim=1, keepdim=True) * threshold
            backward_threshold = torch.mean(
                gt2pred, dim=1, keepdim=True) * threshold
            # only care about distance within threshold (ignore strong outliers)
            pred2gt = torch.where(
                pred2gt < forward_threshold, pred2gt, torch.zeros_like(pred2gt))
            gt2pred = torch.where(
                gt2pred < backward_threshold, gt2pred, torch.zeros_like(gt2pred))

        # pred2gt is for each element in gt, the closest distance to this element
        pred2gt = torch.mean(pred2gt, dim=1)
        gt2pred = torch.mean(gt2pred, dim=1)
        CD_dist = self.forward_weight * pred2gt + gt2pred
        # CD_dist_norm = CD_dist/radius
        cd_loss = torch.mean(CD_dist)
        return cd_loss


if __name__ == '__main__':
    pc1 = torch.randn([2, 600, 3], dtype=torch.float64,
                      requires_grad=True).cuda()
    pc2 = torch.randn([2, 600, 3], dtype=torch.float64,
                      requires_grad=True).cuda()
    chamfer = ChamferLoss()
    from torch.autograd import gradcheck
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    pc2 = pc2.detach()
    test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    print(test)
