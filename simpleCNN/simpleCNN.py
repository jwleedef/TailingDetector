import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3,stride=1)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(7290, 1000)
        # self.fc1 = nn.Linear(7840, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(self.conv2(x)), 2)

        # x = F.relu(self.conv1(x), 2)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        tmp = torch.flatten(x, start_dim=1)
        self.nodeSize = tmp.shape[1]

        x = x.view(-1, self.nodeSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)

# 224 - 28090
# 120 - 7290