'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar

NUM_FLAG = 8


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--name', type=str, help='')
parser.add_argument('--passing_prob', default=[0.0,0.0,0.0,0.0], type=float, nargs="+", help='a')
parser.add_argument('--recycling_prob', default=[0.0,0.0,0.0,0.0], type=float, nargs="+", help='a')
parser.add_argument('--ensemble_size', default=10, type=int, help='a')
parser.add_argument('--lbd', default=0.1, type=float, help='a')
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
cwd = os.getcwd()
## Select Root
train_root = cwd + '/train_1'
test_root = cwd + '/test_1'
###
transform_train = transforms.Compose([
    transforms.Resize((30,30),),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((30,30),),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(root= train_root , transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=30, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root= test_root , transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=30, shuffle=False, num_workers=4)

classes = trainset.classes

# Model
print('==> Building model..')
# net = VGG('VGG19')
#full_net = pResNet18()
full_net = prResNet18()
#net = pResNet18(passing_probs=args.passing_prob)
net = prResNet18(passing_probs=args.passing_prob, recycling_probs=args.recycling_prob)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
full_net = full_net.to(device)
net = net.to(device)
if device == 'cuda':
    full_net = torch.nn.DataParallel(full_net)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir(args.name), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./'+args.name+'/ckpt.pth')
full_net.load_state_dict(checkpoint['net'])
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
# net.module.set_passing_prob(args.passing_prob)
# net.module.set_recycling_prob(args.recycling_prob)

criterion = nn.CrossEntropyLoss()

def online_test(epoch, lbd):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    correct_count = torch.zeros(3**NUM_FLAG)
    entire_count = torch.ones(3**NUM_FLAG)
    szs = 2*NUM_FLAG*torch.ones(3**NUM_FLAG)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_list = []
            flags_list = []
            flags2_list = []
            
            for i in range(args.ensemble_size):
                outputs_temp, flags_temp, flags2_temp = net(inputs)
                outputs_list.append(outputs_temp)
                flags_list.append(flags_temp)
                flags2_list.append(flags2_temp)
                
                if i < 1:
                    outputs = outputs_temp
                else:
                    outputs += outputs_temp
            
            _, predicted = outputs.max(1)
            
            for i in range(args.ensemble_size):
                _, predicted_temp = outputs_list[i].max(1)
                correct_temp = predicted_temp.eq(predicted).sum().item()
                
                # flag encoding
                idx = 0
                sz = 0
                for j in range(NUM_FLAG):
                    if flags_list[i][j] and flags2_list[i][j]:
                        idx += ((3**j)*2)
                        sz += 2
                    elif (flags_list[i][j]) and (not flags2_list[i][j]):
                        idx += ((3**j))
                        sz += 1
                correct_count[idx] += correct_temp
                entire_count[idx] += predicted.size(0)
                szs[idx] = sz
                            
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            #print((correct_count / entire_count).unique(sorted=False))
        
    # best flag decoding
    acc = correct_count / entire_count
    score = acc + lbd*(1 - szs / szs.max())
    sort_val, sort_idx = score.sort(0, descending=True)
    idx = sort_idx[0]
    best_flag = []
    best_flag2 = []
    for j in range(NUM_FLAG):
        if (torch.remainder(idx,3) >= 2) and (torch.remainder(idx,3) < 3) :
            best_flag.append(True)
            best_flag2.append(True)
        elif torch.remainder(idx,3) == 1:
            best_flag.append(True)
            best_flag2.append(False)
        else:
            best_flag.append(False)
            best_flag2.append(False)
        idx_temp = idx / 3
        idx = idx_temp
        
    #print(sort_val.tolist())
    print(best_flag)
    print(best_flag2)
    
    # Print Test Result
    print('Ensemble Test: Epoch#%02d : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # full model testing
    full_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    entire_time = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_list = []
            flags_list = []
            start_time = time.time()
            outputs, _, _ = full_net(inputs)
            entire_time += (time.time() - start_time)
            
            _, predicted = outputs.max(1)            
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    print('Full Model Test: Epoch#%02d : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('running time for each batch : %.8f' % (entire_time/(batch_idx+1)))
    
    # flag setting
    full_net.module.flags = best_flag
    full_net.module.flags2 = best_flag2
    full_net.module.cum_num_blocks = [2,4,6,8]
    full_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    entire_time = 0
    
    # flagged model testing
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_list = []
            flags_list = []
            
            start_time = time.time()
            outputs, _, _ = full_net(inputs)
            entire_time += (time.time() - start_time)
            
            _, predicted = outputs.max(1)            
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    print('Adapted Model Test: Epoch#%02d : Loss: %.3f | Acc: %.8f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('running time for each batch : %.8f' % (entire_time/(batch_idx+1)))

online_test(0, args.lbd)
