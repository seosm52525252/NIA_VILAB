'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, passing_prob=0.0, index=0):
        super(BasicBlock, self).__init__()
        self.index = index
        self.half_in_planes = in_planes//2
        self.passing_prob = passing_prob
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, inputs):
        
        x = inputs[0]
        flag = inputs[1]
        test_flag = inputs[2]
                
        #if self.training:
        if (random.random() < self.passing_prob) or (not test_flag[self.index]):
            out = self.shortcut(x)
            flag.append(False)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            flag.append(True)
        
        return (out, flag, test_flag)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, scale=10.0):
        super(Bottleneck, self).__init__()
        self.half_in_planes = in_planes//2
        self.scale = scale
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))        
        sc1 = self.scale*torch.mean(x[:,:(self.half_in_planes),:,:], dim=(1,2,3), keepdim=True)
        sc2 = self.scale*torch.mean(x[:,(self.half_in_planes):,:,:], dim=(1,2,3), keepdim=True)
        sc = torch.exp(sc1) / (torch.exp(sc1) + torch.exp(sc2))
        out = sc*out + (1-sc)*self.shortcut(x)
        out = F.relu(out)
        return out


class pResNet(nn.Module):
    def __init__(self, block, num_blocks, passing_probs=[0.0,0.0,0.0,0.0], num_classes=10):
        super(pResNet, self).__init__()
        self.in_planes = 64
        self.flags = [True, True, True, True, True, True, True, True]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, passing_prob=passing_probs[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, passing_prob=passing_probs[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, passing_prob=passing_probs[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, passing_prob=passing_probs[3])
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, passing_prob):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, passing_prob, i))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1((out, [], self.flags[0:2]))
        out2 = self.layer2((out2[0], out2[1], self.flags[2:4]))
        out2 = self.layer3((out2[0], out2[1], self.flags[4:6]))
        out2 = self.layer4((out2[0], out2[1], self.flags[6:8]))
        flags = out2[1]
        out = F.avg_pool2d(out2[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, flags
    
def pResNet18(passing_probs=[0.0, 0.0, 0.0, 0.0]):
    return pResNet(BasicBlock, [2, 2, 2, 2], passing_probs=passing_probs)
    

def pResNet34():
    return pResNet(BasicBlock, [3, 4, 6, 3])


def pResNet50():
    return pResNet(Bottleneck, [3, 4, 6, 3])


def pResNet101():
    return pResNet(Bottleneck, [3, 4, 23, 3])


def pResNet152():
    return pResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
