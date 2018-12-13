import torch

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

    def forward(self):
        pass
