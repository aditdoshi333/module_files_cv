import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models.resnet import *
import os


def get_data_loader(train_set, test_set, batch_size=128):

    train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def get_model(norm_type="BatchNorm",model_name="resnet_18"):

    if model_name == "resnet_18":
        net = ResNet18(norm_type=norm_type)

    return net



def set_model_train_config(model, loss="cross_entropy", optimizer="SGD", lr=0.1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    model = model.to(device)

    if loss=="cross_entropy":
        criterion = nn.CrossEntropyLoss()
    
    if optimizer=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return device, classes, model, criterion, optimizer, scheduler


def train(net, criterion, optimizer, device, trainloader, train_losses, train_acc):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_losses.append(train_loss/(batch_idx+1))
    train_acc.append(100.*correct/total)
    return train_losses, train_acc


def test(net, criterion, device, testloader, test_losses, test_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    test_losses.append(test_loss/(batch_idx+1))
    test_acc.append(100.*correct/total)
    
    
    return test_losses, test_acc
    




def training_loop(no_of_epoch, net, criterion, optimizer, device, trainloader, testloader, scheduler):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(no_of_epoch):
        print(f"EPOCH : {epoch}")
        print()
        train_loss, train_acc = train(net, criterion, optimizer, device, trainloader, train_loss, train_acc)
        test_loss, test_acc_list = test(net, criterion, device, testloader, test_loss, test_acc)
        scheduler.step(test_loss[-1])
    return train_loss, train_acc, test_loss, test_acc_list
