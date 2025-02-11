import math

from torch import nn

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature, double_layer=False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        if double_layer:
            self.Connectors = nn.Sequential(
                nn.Conv2d(in_feature, out_feature/2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feature/2),
                nn.SiLU(),
                nn.Conv2d(out_feature/2, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feature),
                nn.SiLU(),
            )
        else:
            self.Connectors = nn.Sequential(
                nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feature), nn.ReLU())

        # network params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.samples.fill_(1)
                m.bias.samples.zero_()

    def forward(self, student):
        student = self.Connectors(student)
        return student

class statm_loss(nn.Module):
    def __init__(self, eps=2):
        super(statm_loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        x = x.view(x.size(0), x.size(1), -1)
        y = y.view(y.size(0), y.size(1), -1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean - y_mean).pow(2).mean(1)
        return mean_gap.mean()