import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import random

is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

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

def fit(epoch,model,phase='train',volatile=False):
    if phase == 'train':
        model.train()
    if phase == 'valid':
        model.eval()
        volatile=True

    running_loss = 0.0
    running_correct = 0

    for inputs, labels in dataloaders[phase]:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        
        if phase == 'train':
            optimizer.zero_grad()
        output = model(inputs)

        # print(f'\n[OUTPUT] : {output} {output.size()}')
        # print(f'[LABEL] : {labels} \n')

        loss = F.nll_loss(output, labels)

        if phase == 'train':
            loss.backward()
            optimizer.step()
        
    _, preds = torch.max(output, 1)

    accuracy = 0
    return loss,accuracy

def torchLoader(path):
    ret = torch.load(path)
    return ret

#########################
# Data Load
#########################

from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

data_path = '../Dataset/train'
train_dataset = datasets.DatasetFolder(root=data_path, loader=torchLoader, extensions='.pt')

batch_size  = 128
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

 ########################
 # Main 
 ########################

learning_rate = 0.01

model = Net()
if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch,model,phase='train')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,phase='valid')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    print(f'epoch : {epoch}')
    print(f'train loss : {epoch_loss}')
    print(f'valid loss : {val_epoch_loss}')
torch.save(model, 'model.pt')