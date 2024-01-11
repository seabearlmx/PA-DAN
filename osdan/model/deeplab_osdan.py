import torch
import torch.nn as nn
import torch.nn.functional as F

from osdan.model.deeplabv2 import Bottleneck, ResNetMulti

AFFINE_PAR = True


class OsdanResNet(ResNetMulti):
    def __init__(self, block, layers, num_classes, multi_level):
        super().__init__(block, layers, num_classes, multi_level)
        self.enc4_1 = nn.Conv2d(19, 128, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.enc4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.enc4_3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.enc4_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_5 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=True)

        self.interp_target = nn.Upsample(size=(512, 1024), mode='bilinear',
                                         align_corners=True)

        self.enc4_1.weight.data.normal_(0, 0.01)
        self.enc4_2.weight.data.normal_(0, 0.01)
        self.enc4_3.weight.data.normal_(0, 0.01)
        self.enc4_4.weight.data.normal_(0, 0.01)
        self.enc4_5.weight.data.normal_(0, 0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4 = self.layer4(x)
        seg_conv5 = self.layer6(x4)  # produce segmap 2

        seg_per = self.interp_target(seg_conv5)
        x4_enc = self.enc4_1(seg_per)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.maxpool(x4_enc)
        x4_enc = self.enc4_2(x4_enc)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_3(x4_enc)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_4(x4_enc)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_5(x4_enc)
        syth_struct = x4_enc

        mix_feature = x4 + syth_struct
        struct_seg = self.layer5(mix_feature)

        return syth_struct, struct_seg, seg_conv5

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        b.append(self.enc4_1.parameters())
        b.append(self.enc4_2.parameters())
        b.append(self.enc4_3.parameters())
        b.append(self.enc4_4.parameters())
        b.append(self.enc4_5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


def get_osdan_deeplab(num_classes=16, multi_level=False):
    model = OsdanResNet(
        Bottleneck, [3, 4, 23, 3], num_classes, multi_level
    )
    return model
