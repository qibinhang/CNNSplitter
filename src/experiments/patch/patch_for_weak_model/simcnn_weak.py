import math
import torch
import torch.nn as nn


class SimCNNWeak(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(SimCNNWeak, self).__init__()
        self.num_classes = num_classes
        is_modular = True
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 64), (64, 64), (64, 64)]
            is_modular = False

        # the name of conv layer must be 'conv_*'
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            setattr(self, f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))

        self.fc_4 = nn.Linear(conv_configs[-1][-1] * 4, num_classes)

        if not is_modular:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        y = torch.relu(self.conv_0(x))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_1(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_2(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_3(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = y.view(y.size(0), -1)

        pred = self.fc_4(y)
        return pred
