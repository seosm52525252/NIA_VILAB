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

    def __init__(self, in_planes, planes, stride=1, passing_prob=0.0, recycling_prob=0.0, index=0):
        super(BasicBlock, self).__init__()
        self.index = index
        self.half_in_planes = in_planes//2
        self.passing_prob = passing_prob
        self.recycling_prob = recycling_prob
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
        
    def forward(self, x2):
        
        x = x2[0]
        prev_layer = x2[1]
        flag = x2[2]
        flag2 = x2[3]
        test_flag = x2[4]        
        test_flag2 = x2[5]
        
        #if self.training:
        curr_layer = prev_layer

        rand_val = random.random()
        
        # skip connection only
        if (rand_val < self.passing_prob) or ((not test_flag[self.index]) and (not test_flag2[self.index])):
            out = self.shortcut(x)
            flag.append(False)
            flag2.append(False)
            return (out, curr_layer, flag, flag2, test_flag, test_flag2)
        # recycling block
        elif (rand_val < self.passing_prob + self.recycling_prob) or ((test_flag[self.index]) and (not test_flag2[self.index])):
            out2 = prev_layer((x, curr_layer, flag, flag2, [True]*(prev_layer.index+1), [True]*(prev_layer.index+1)))
            flag.append(True)
            flag2.append(False)
            return (out2[0], curr_layer, flag, flag2, test_flag, test_flag2)
        # original estimation
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            
            curr_layer = self
            flag.append(True)
            flag2.append(True)
            return (out, curr_layer, flag, flag2, test_flag, test_flag2)


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


class prResNet(nn.Module):
    def __init__(self, block, num_blocks, passing_probs=[0.0,0.0,0.0,0.0], recycling_probs=[0.0,0.0,0.0,0.0], num_classes=8):
        super(prResNet, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = 256
        self.flags = [True]*sum(num_blocks)
        self.flags2 = [True]*sum(num_blocks)

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=1, passing_prob=passing_probs[0], recycling_prob=recycling_probs[0])
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, passing_prob=passing_probs[1], recycling_prob=recycling_probs[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, passing_prob=passing_probs[2], recycling_prob=recycling_probs[2])
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2, passing_prob=passing_probs[3], recycling_prob=recycling_probs[3])
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, passing_prob, recycling_prob):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, passing_prob, recycling_prob, i))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        curr_layer = None
        out = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1((out, curr_layer, [], [], self.flags[:(self.num_blocks[0])], self.flags2[:(self.num_blocks[0])]))
        out2 = self.layer2((out2[0], out2[1], out2[2], out2[3], self.flags[(self.num_blocks[0]):(sum(self.num_blocks[0:2]))], self.flags2[(self.num_blocks[0]):(sum(self.num_blocks[0:2]))]))
        out2 = self.layer3((out2[0], out2[1], out2[2], out2[3], self.flags[(sum(self.num_blocks[0:2])):(sum(self.num_blocks[0:3]))], self.flags2[(sum(self.num_blocks[0:2])):(sum(self.num_blocks[0:3]))]))
        out2 = self.layer4((out2[0], out2[1], out2[2], out2[3], self.flags[(sum(self.num_blocks[0:3])):], self.flags2[(sum(self.num_blocks[0:3])):]))
#         out2 = self.layer1((out, curr_layer, [], [], self.flags[:2], self.flags2[:2]))
#         out2 = self.layer2((out2[0], out2[1], out2[2], out2[3], self.flags[2:4], self.flags2[2:4]))
#         out2 = self.layer3((out2[0], out2[1], out2[2], out2[3], self.flags[4:6], self.flags2[4:6]))
#         out2 = self.layer4((out2[0], out2[1], out2[2], out2[3], self.flags[6:8], self.flags2[6:8]))
#         out2 = self.layer1((out, curr_layer, [], [], self.flags[0:self.cum_num_blocks[0]], self.flags2[0:self.cum_num_blocks[0]]))
#         out2 = self.layer2((out2[0], out2[1], out2[2], out2[3], self.flags[self.cum_num_blocks[0]:self.cum_num_blocks[1]], self.flags2[self.cum_num_blocks[0]:self.cum_num_blocks[1]]))
#         out2 = self.layer3((out2[0], out2[1], out2[2], out2[3], self.flags[self.cum_num_blocks[1]:self.cum_num_blocks[2]], self.flags2[self.cum_num_blocks[1]:self.cum_num_blocks[2]]))
#         out2 = self.layer4((out2[0], out2[1], out2[2], out2[3], self.flags[self.cum_num_blocks[2]:self.cum_num_blocks[3]], self.flags2[self.cum_num_blocks[2]:self.cum_num_blocks[3]]))    
    
        flag = out2[2]
        flag2 = out2[3]
        
        out = F.avg_pool2d(out2[0], out2[0].size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return (out, flag, flag2)
    
    def set_passing_prob(self, passing_probs):
        
        if not (len(passing_probs) == 4):
            return False
        
        self.layer1.passing_prob = passing_probs[0]
        self.layer2.passing_prob = passing_probs[1]
        self.layer3.passing_prob = passing_probs[2]
        self.layer4.passing_prob = passing_probs[3]
        
        return True
    
    def set_recycling_prob(self, recycling_probs):
        
        if not (len(recycling_probs) == 4):
            return False
        
        self.layer1.recycling_prob = recycling_probs[0]
        self.layer2.recycling_prob = recycling_probs[1]
        self.layer3.recycling_prob = recycling_probs[2]
        self.layer4.recycling_prob = recycling_probs[3]
        
        return True


def prResNet18(passing_probs=[0.0, 0.0, 0.0, 0.0], recycling_probs=[0.0, 0.0, 0.0, 0.0]):
    return prResNet(BasicBlock, [2, 2, 2, 2], passing_probs=passing_probs, recycling_probs=recycling_probs)
    

def prResNet34():
    return prResNet(BasicBlock, [3, 4, 6, 3])


def prResNet50():
    return prResNet(Bottleneck, [3, 4, 6, 3])


def prResNet101():
    return prResNet(Bottleneck, [3, 4, 23, 3])


def prResNet152():
    return prResNet(Bottleneck, [3, 8, 36, 3])

