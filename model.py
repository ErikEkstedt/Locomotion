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


# ------------------
def testing():
    batch_size=32
    seq_len = 5
    joints = 63
    x = torch.rand(batch_size, seq_len, joints)

    in_channels = seq_len
    out_channels = 10
    kernel_size = 5
    stride = 2
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    out = m(x)

if __name__ == "__main__":
    testing()
