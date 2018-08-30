import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 in_channels=20,
                 mid_channels=[20, 10],
                 out_size=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels[0])
        self.conv2 = nn.Conv1d(mid_channels[0], mid_channels[1])
        self.conv3 = nn.Conv1d(mid_channels[1], mid_channels[2])
        self.out = nn.Linear(mid_channels[2], out_size=10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        return self.out(x)


class MLP(nn.Module):
    def __init__(self,
                 in_size=126,
                 hidden=128,
                 out_size=63):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_size)

        self.activation = F.elu

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.out(x).reshape(bs, -1)

