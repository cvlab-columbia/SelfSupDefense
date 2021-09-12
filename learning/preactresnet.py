'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNetSSL(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetSSL, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        return out, hidden

class PreActResNetMhead(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetMhead, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out5 = F.relu(self.bn(out5))
        out6 = F.avg_pool2d(out5, 4)
        out6 = out6.view(out6.size(0), -1)
        out = self.linear(out6)
        return out, out3, out4, out5, out6


class PreActResNetBD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetBD, self).__init__()
        self.in_planes = 64

        BDbandwidth = 3
        self.conv1bd = nn.Conv2d(3, BDbandwidth, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        # self.conv2bd = nn.Conv2d(3, BDbandwidth, kernel_size=3, stride=1,
        #                          padding=1, bias=False)
        # self.conv3bd = nn.Conv2d(3, BDbandwidth, kernel_size=3, stride=2,
        #                          padding=1, bias=False)

        self.BDbandwidth=BDbandwidth

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, BD=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, BD=False)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, BD=False)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, BD=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for cnt, stride in enumerate(strides):
            layers.append(block(self.in_planes + self.BDbandwidth if BD and cnt==0 else self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, backdoor):
        bd1 = self.conv1bd(backdoor)
        # bd2 = self.conv2bd(backdoor)
        # bd3 = self.conv3bd(backdoor)

        out = self.conv1(x)
        out = torch.cat((out, bd1), dim=1)
        out = self.layer1(out)
        # out = torch.cat((out, bd2), dim=1)
        out = self.layer2(out)
        # out = torch.cat((out, bd3), dim=1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNetMed(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, constrastiv_head=16):
        super(PreActResNetMed, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.linear2 = nn.Linear(512*block.expansion, 512)
        self.linear3 = nn.Linear(512, 16)
        self.bn2 = nn.BatchNorm1d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.debn1 = nn.BatchNorm2d(256)
        self.debn2 = nn.BatchNorm2d(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_detach=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))

        # import pdb; pdb.set_trace()

        de = F.relu(self.debn1(self.deconv1(out)))
        de = F.relu(self.debn2(self.deconv2(de)))
        de = self.deconv3(de)

        out = F.avg_pool2d(out, 4)
        hidden_out = out.view(out.size(0), -1)
        out = self.linear(hidden_out)

        if use_detach:
            hidden_out_2 = hidden_out.detach()
        else:
            hidden_out_2 = hidden_out
        head = F.relu(self.bn2(self.linear2(hidden_out_2)))
        head = self.linear3(head)
        return out, (head, de)


class PreActResNetMedL(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetMedL, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.linear2 = nn.Linear(512*block.expansion, 8)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.convd2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.convd3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.debn1 = nn.BatchNorm2d(256)
        self.debn11 = nn.BatchNorm2d(256)
        self.debn2 = nn.BatchNorm2d(128)
        self.debn22 = nn.BatchNorm2d(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))

        # import pdb; pdb.set_trace()

        de = F.relu(self.debn1(self.deconv1(out)))
        de = F.relu(self.debn11(self.convd2(de)))
        de = F.relu(self.debn2(self.deconv2(de)))
        de = F.relu(self.debn22(self.convd3(de)))
        de = self.deconv3(de)

        out = F.avg_pool2d(out, 4)
        hidden_out = out.view(out.size(0), -1)
        out = self.linear(hidden_out)
        head = self.linear2(hidden_out)
        return out, (head, de)

head_num=8 #
print('head num', head_num)

class Res18_out3_model(nn.Module):
    def __init__(self):
        super(Res18_out3_model, self).__init__()

        self.convd1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.convd2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)

        self.bn3 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, head_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.convd1(x)))
        x = F.relu(self.bn2(self.convd2(x)))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x

class Res18_out4_model(nn.Module):
    def __init__(self):
        super(Res18_out4_model, self).__init__()

        self.convd2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(512)

        self.bn3 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, head_num)

    def forward(self, x):
        x = F.relu(self.bn2(self.convd2(x)))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


class Res18_out5_model(nn.Module): # right before the average pooling layer
    def __init__(self):
        super(Res18_out5_model, self).__init__()

        self.convd2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False) #2*2

        self.bn2 = nn.BatchNorm2d(512)

        self.bn3 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(512*4, 128)
        self.linear2 = nn.Linear(128, head_num)

    def forward(self, x):
        x = F.relu(self.bn2(self.convd2(x)))
        # import pdb; pdb.set_trace()
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


class Res18_out6_model(nn.Module): # after the avg pooling layer
    def __init__(self):
        super(Res18_out6_model, self).__init__()

        self.bn3 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, head_num)

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


class Res50_ssl_model(nn.Module): # after the avg pooling layer
    def __init__(self):
        super(Res50_ssl_model, self).__init__()

        self.bn3 = nn.BatchNorm1d(512)
        self.linear = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18SSL(num_classes=10):
    return PreActResNetSSL(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18Mhead(num_classes=10):
    return PreActResNetMhead(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18BD(num_classes=10):
    return PreActResNetBD(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18Med(num_classes=10, constrastiv_head=16):
    return PreActResNetMed(PreActBlock, [2,2,2,2], num_classes=num_classes, constrastiv_head=constrastiv_head)

def PreActResNet18MedL(num_classes=10):
    return PreActResNetMedL(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
