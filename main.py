import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import MocapDataset, get_dataloaders
from models.vq_vae import Model
from tensorboardX import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()

num_epochs = 100
batch_size = 32
in_channels = 1
out_channels = in_channels
num_training_updates = 25000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dset = MocapDataset()

train_loader, test_loader = get_dataloaders(batch_size=batch_size)

print('There are {} minibatches with {} batch_size in one epoch'.format(len(train_loader), train_loader.batch_size))

model = Model(in_channels,
              out_channels,
              num_hiddens=128,
              num_residual_layers=32,
              num_residual_hiddens=2,
              embedding_dim=64,
              num_embeddings=512,
              commitment_cost=commitment_cost,
              decay=decay).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
loss_function = nn.MSELoss()

# Data variance for normalizing
# train_data.shape
# torch.Size([1835, 240, 66])
# train_data = train_loader.dataset.data  # torch.size([1
# train_data_mean = train_data.mean(dim=1).mean(dim=0)
# train_data_var = train_data.var(dim=1).mean(dim=0).pow(2)
# train_data_standard = (train_data-train_data_mean)/train_data_var


for epoch in range(1, num_epochs+1):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        j, c = batch['coord'], batch['ctrl']  # (Bx240x63), (B, 240, 3)
        data = torch.cat((j,c), dim=2).to(device)
        data = data.unsqueeze(1)  # Add dummy "image" channel
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = torch.mean((data_recon - data)**2)
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar('Loss', loss.item(), i)
        writer.add_scalar('Reconstruction Error', recon_error.item(), i)
        writer.add_scalar('Perplexity', perplexity.item(), i)
    writer.add_scalar('Epoch Loss', epoch_loss.item(), i)
    torch.save('checkpoints/model_epoch_{}_loss_{}.pt'.format(epoch, epoch_loss.item()), model)
