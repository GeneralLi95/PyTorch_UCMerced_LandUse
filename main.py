import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim import Adam

import os
import argparse

from models import *
from utils import get_progress_bar, update_progress_bar, ApplyTransform

# 0. Define some parameters
parser = argparse.ArgumentParser(description='UCMerced Land Use')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load and normalizing dataset
# 1. Data augmentation
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

total_dataset = datasets.ImageFolder('Images', transform=None)
train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size

train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

train_dataset = ApplyTransform(train_dataset, transform=transforms_train)
test_dataset = ApplyTransform(test_dataset, transform=transforms_test)

train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=16)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=16)

# 2. Define a Convolutional Network



# net, model_name = LeNet(), 'LeNet'
# net, model_name = ResNet18(), 'ResNet18'
net, model_name = ResNet34(), 'ResNet34'

print(model_name + ' is ready!')

net = net.to(device)
# Use GPU or not
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs")
    cudnn.benchmark = True

start_epoch = 0
best_acc = 0

if args.resume == True:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/' + model_name), 'Error : no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + model_name + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

# 3. Define a loss function
criterion = nn.CrossEntropyLoss()
# optimizer = Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# 4. Train the network on the training data

def train(epoch):
    running_loss = 0.0
    net.train()
    correct = 0
    total = 0
    progress_bar_obj = get_progress_bar(len(train_dataset_loader))
    print('Epoch', epoch, 'Train')
    for i, (inputs, labels) in enumerate(train_dataset_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # this line doesn't work when use cpu
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        update_progress_bar(progress_bar_obj, index=i, loss=(running_loss / (i + 1)), acc=100. * (correct / total),
                            c=correct, t=total)


# 5. Test Network
def test(epoch):
    global best_acc
    net.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataset_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print()
    print("Accuracy of whole test dataset: %.2f %%" % acc)

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/' + model_name):
            os.mkdir('checkpoint/' + model_name)
        torch.save(state, './checkpoint/' + model_name + '/ckpt.pth')
        best_acc = acc
        print('Acc > best_acc, Saving net, acc')


for epoch in range(start_epoch, start_epoch + 150):
    train(epoch)
    test(epoch)
