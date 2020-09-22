import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable

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

def fit(epoch,model,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True

    running_loss = 0.0
    running_correct = 0
    
    data = torch.load("/home/jwlee/kaia/EdgeAnalysisModule/testData-03.pt")
    data = data.cuda()
    data = Variable(data, volatile)
    
    print(f'\n[DATA] : {data.sum()} {data.size()}')

    target = torch.tensor([1])
    target = target.cuda()
    target = Variable(target)

    if phase == 'training':
        optimizer.zero_grad()
    output = model(data)
    # output = output.view(-1)

    print(f'\n[OUTPUT] : {output} {output.size()}')
    print(f'[TARGET] : {target}')

    loss = F.nll_loss(output, target)
    # criterion = nn.BCELoss()
    # loss = criterion(output, target)

    running_loss += F.nll_loss(output, target).data
    preds = output.data.max(dim=1,keepdim=True)[1]
    running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

    print(f'[PREDICTION] : {preds}\n')

    if phase == 'training':
        loss.backward()
        optimizer.step()
    
    loss = running_loss.item()#/len(data_loader.dataset)
    accuracy = 100.0 * running_correct.item()#/len(data_loader.dataset)
    
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{1} {accuracy:{10}.{4}}')
    return loss,accuracy


learning_rate = 0.01

model = Net()
if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

# train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

model = torch.load('model.pt')
epoch = 1
epoch_loss , epoch_accuracy = fit(epoch,model,phase='validation')

print(f'loss : {epoch_loss}')
print(f'accuracy : {epoch_accuracy}')