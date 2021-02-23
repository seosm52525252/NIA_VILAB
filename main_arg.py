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

from models import *
#from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--name', default='checkpoint', type=str, help='a')
parser.add_argument('--passing_prob', default=[0.0,0.0,0.0,0.0], type=float, nargs="+", help='a')
parser.add_argument('--recycling_prob', default=[0.0,0.0,0.0,0.0], type=float, nargs="+", help='a')
args = parser.parse_args()

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
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
#    print('\nEpoch: %d' % epoch)

    if epoch > 160:
        optimizer = optim.SGD(net.parameters(), lr=args.lr*0.1, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs[0]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    print('Training: Epoch#%02d : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = outputs[0]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Print Test Result
    print('Test: Epoch#%02d : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.name):
            os.mkdir(args.name)
        torch.save(state, './'+args.name+'/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
test(epoch)
