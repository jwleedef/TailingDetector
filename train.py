import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
import random
import argparse

from simpleCNN.simpleCNN import Net

def train(epoch, model, dataloaders, batch_num, phase='train', volatile=False):
    if phase == 'train':
        model.train()
    if phase == 'valid':
        model.eval()
        volatile=True

    running_loss = 0.0
    running_correct = 0

    for inputs, labels in dataloaders[phase]:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile), Variable(labels)
        
        if phase == 'train':
            optimizer.zero_grad()
        output = model(inputs)

        loss = F.nll_loss(output, labels)
        # print(loss)
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
    return loss

def torchLoader(path):
    ret = torch.load(path)
    return ret

def dataload(batchSize, dataPath):
    train_dataset = datasets.DatasetFolder(root=dataPath, loader=torchLoader, extensions='.pt')

    random_seed = 555
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=random_seed)
    
    datasets = {}
    datasets['train'] = Subset(train_dataset, train_idx)
    datasets['valid'] = Subset(train_dataset, val_idx)

    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=4)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=4)
    batch_num['train'], batch_num['valid'], = len(dataloaders['train']), len(dataloaders['valid'])

    return dataloaders, batch_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Dataset/train', help='path to train data directory')
    parser.add_argument('--batchsize', type=int, default=512, help='train batchsize')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    args = parser.parse_args()

    dataPath = args.data_path
    batchSize = args.batchsize
    learning_rate = args.learning_rate
    epoch = args.epoch

    is_cuda=False
    if torch.cuda.is_available():
        is_cuda = True

    model = Net()
    if is_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = optim.SGD(model.parameters(),lr=learning_rate)

    dataloaders, batch_num = dataload(batchSize, dataPath)

    train_losses , val_losses = [],[]

    for index in range(epoch):
        epoch_loss = train(index, model, dataloaders, batch_num, phase='train')
        val_epoch_loss  = train(index, model, dataloaders, batch_num, phase='valid')
        train_losses.append(epoch_loss)
        print(f'epoch : {index}')
        print(f'train loss : {epoch_loss}')
        print(f'valid loss : {val_epoch_loss}')
        torch.save(model, 'model.pt')
        print(f'[UPDATE] model.pt ')