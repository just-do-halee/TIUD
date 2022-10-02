import torch
import torch.nn as nn

class APReLU(nn.Module):
    def __init__(self, batch_norm, a=10, eps=1e-8):
        super(APReLU, self).__init__()
        self.a = a
        self.eps = eps
        self.batch_norm = batch_norm
        self.analysis1 = nn.Sequential(
            nn.Linear(1, a), 
            nn.Linear(a, 1),
            nn.Sigmoid()
        )
        self.analysis2 = nn.Sequential(
            nn.Linear(1, a),
            nn.Linear(a, 1),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_parameter('bias', self.bias)

    def gate(self):
        active = self.analysis1(self.bias)
        inhibit = self.analysis2(active)
        return active, inhibit

    def sep(self, x, a, b):
        return torch.where((x>=0), a*x, b*x)

    def forward(self, x):
        x = self.batch_norm(x)

        active, inhibit = self.gate()

        x = self.sep(x, active, (1-active)) / (inhibit + self.eps)

        return x


class APReLU1d(APReLU):
    def __init__(self, out_channels, a=10, eps=1e-8):
        super(APReLU1d, self).__init__(nn.BatchNorm1d(out_channels, eps=eps, affine=True), a, eps)

class APReLU2d(APReLU):
    def __init__(self, out_channels, a=10, eps=1e-8):
        super(APReLU2d, self).__init__(nn.BatchNorm2d(out_channels, eps=eps, affine=True), a, eps)
