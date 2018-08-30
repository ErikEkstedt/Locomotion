import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import MocapDataset, get_dataloaders
from model import MLP


# dset = MocapDataset()

train_loader, test_loader = get_dataloaders()

print('There are {} minibatches with {} batch_size in one epoch'.format(len(train_loader), train_loader.batch_size))

model = MLP(in_size=126, hidden=128, out_size=63)
optimizer =  optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.MSELoss()

n_epochs = 10 
for epoch in range(1, n_epochs +1):
    epoch_loss = 0
    for d in tqdm(train_loader):
        j_T, c_T = d['j_pos'], d['ctrl']
        batch_size = j.shape[0]
        T = j_T.shape[1]
        j_prev = j_T[:, 0, :]
        c_prev = c_T[:, 0, :]
        batch_loss = 0
        for t in range(1, T-1):
            j_current = j_T[:, t, :]
            c_current = c_T[:, t, :]
            j_target = j_T[:, t+1, :]
            j = torch.stack((j_current, j_prev), dim=1)
            c = torch.stack((c_current, c_prev), dim=1)
            out = model(j)
            loss = loss_function(out, j_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j_prev = j_current
            c_prev = c_current
            batch_loss += loss.item()
        epoch_loss += batch_loss
    print('Epoch {}/{} loss: {}'.format(epoch, n_epochs, epoch_loss ))






