import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

import time
import copy
import random
import argparse

from simpleCNN.simpleCNN import Net
from simpleUnionCNN.simpleUnionCNN import UnionNet
from efficientnet import EfficientNet

def train(model, modelName, dataloaders, batch_num, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:            
            if phase == 'train':
                since = time.time()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                end = time.time()
                print(f'trained {int(end - since)} seconds')                
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))
           
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                model.load_state_dict(best_model_wts)

        model.load_state_dict(best_model_wts)
        torch.save(model, f'weights/{modelName}/{modelName}-{epoch}-model.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, f'{modelName}-best-model.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

def torchLoader(path):
    ret = torch.load(path)
    return ret

def dataload(batchSize, dataPath):
    train_dataset = datasets.DatasetFolder(root=dataPath, loader=torchLoader, extensions='.pt')

    random_seed = 555
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=random_seed)
    
    dataset = {}
    dataset['train'] = Subset(train_dataset, train_idx)
    dataset['valid'] = Subset(train_dataset, val_idx)

    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                batch_size=batchSize, shuffle=True,
                                                num_workers=4)
    dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                batch_size=batchSize, shuffle=False,
                                                num_workers=4)
    batch_num['train'], batch_num['valid'], = len(dataloaders['train']), len(dataloaders['valid'])

    return dataloaders, batch_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simpleCNN', help='select model : [simpleCNN, simpleUnionCNN, efficientNet]')
    parser.add_argument('--data_path', type=str, default='Dataset/train', help='path to train data directory')
    parser.add_argument('--batchsize', type=int, default=32, help='train batchsize')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    args = parser.parse_args()

    model = args.model
    modelName = model
    dataPath = args.data_path
    batchSize = args.batchsize
    learning_rate = args.learning_rate
    epoch = args.epoch

    is_cuda=False
    if torch.cuda.is_available():
        is_cuda = True

    dataloaders, batch_num = dataload(batchSize, dataPath)
    criterion = nn.CrossEntropyLoss()

    if model == 'simpleCNN':
        model = Net()
        if is_cuda:
            model = torch.nn.DataParallel(model)
            model.cuda()

    elif model == 'simpleUnionCNN':
        model = UnionNet()
        if is_cuda:
            model = torch.nn.DataParallel(model)
            model.cuda()        

    elif model == 'efficientNet':
        model_name = 'efficientnet-b0'
        model = EfficientNet.from_name(model_name, num_classes=3)
        if is_cuda:
            model = torch.nn.DataParallel(model)
            model.cuda()

    optimizer_ft = optim.SGD(model.parameters(), 
                            lr = learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)
    lmbda = lambda epoch: 0.98739        
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)                 
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train(model, modelName, dataloaders, batch_num, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)
