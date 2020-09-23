import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import argparse
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 16, kernel_size=5,stride=3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=5,stride=2)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=5,stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(40, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x,p=0.1, training=self.training)
        
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)
        # return x

def torchLoader(path):
    ret = torch.load(path)
    return ret

def infer(model, dataPath, batchSize, is_cuda=False):
    model.eval()
    
    test_dataset = datasets.DatasetFolder(root=dataPath, loader=torchLoader, extensions='.pt')

    batch_size = batchSize
    dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize,
                                            shuffle=False, num_workers=4)
    batch_num = len(dataloaders)

    for inputs, labels in dataloaders:
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # print(f'\n[DATA] {inputs}')

        output = model(inputs)
        print(f'\n[OUTPUT] : {output}')

        preds = output.data.max(dim=1,keepdim=True)[1]
        print(f'[PREDICTION] : {preds}')
        print(f'[LABEL] : {labels}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../Dataset/test', help='Test Data Directory')
    parser.add_argument('--batchsize', type=int, default=1, help='Test Dataset Batch size')
    args = parser.parse_args()

    dataPath = args.data_path
    batchSize = args.batchsize

    is_cuda=False
    if torch.cuda.is_available():
        is_cuda = True
    
    model = torch.load('model.pt')
    if is_cuda:
        model.cuda()

    infer(model, dataPath, batchSize, is_cuda)