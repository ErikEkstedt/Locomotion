from dataset import dataset
from model import Model


seq_len = 20
model = Model(in_channels=seq_len)
dloader = dataset(seq_len=seq_len)

for batch, d in enumerate(dloader):
    j, c, reset = d
    if reset:
        print('BATCH: ', int(batch/(240/60)))
    print(j.shape)
    print(c.shape)
    print(reset)
    out = model(j)
    input()
