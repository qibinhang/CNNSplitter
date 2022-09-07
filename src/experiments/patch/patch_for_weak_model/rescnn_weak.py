import torch.nn as nn


class ResCNNWeak(nn.Module):
    def __init__(self, num_classes=10, block_configs=None):
        super().__init__()
        self.num_classes = num_classes
        if block_configs is None:
            block_configs = self.load_default_conv_configs()

        for i, each_block_config in enumerate(block_configs):
            setattr(self, f'conv_{i}', self.conv_block(*each_block_config))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.4),
                                        nn.Linear(block_configs[-1][1] * 64, num_classes))

    def load_default_conv_configs(self):
        block_configs = [(3, 64, True), (64, 64, False),
                         (64, 64, False), (64, 64, True)]
        return block_configs

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_0(x)

        res = out
        out = self.conv_1(out)
        out = self.conv_2(out) + res

        out = self.conv_3(out)
        out = self.classifier(out)
        return out
