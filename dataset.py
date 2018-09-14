from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# TODO
# Easy validation split


class MocapDataset(Dataset):
    def __init__(self, root='mocap/data', test=False):
        if test:
            dpath = join(root, 'edinburgh_locomotion_test.npz')
        else:
            dpath = join(root, 'edinburgh_locomotion_train.npz')
        self.data = torch.from_numpy(np.load(dpath)['clips']).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions = self.data[idx][:, :-3]
        ctrl = self.data[idx][:, -3:]
        return {'coord': joint_positions, 'ctrl':ctrl}


class MocapSequenceLoader(DataLoader):
    def __init__(self,
                 dset,
                 batch_size,
                 seq_len=1,
                 overlap_len=0,
                 rnn=False,
                 *args,
                 **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.dset = dset
        self.rnn = rnn

    def __iter__(self):
        for batch in super().__iter__():
            jpos = batch['coord']
            ctrl = batch['ctrl']
            batch_size, n_frames, n_joints = jpos.shape

            reset = True
            for i in range(self.overlap_len, n_frames, self.seq_len):
                j = jpos[:, i, :]
                c = ctrl[:, i, :]
                yield j, c, reset
                reset = False


def get_dataloaders(batch_size=32, root='mocap/data', *args, **kwargs):
    train_dset = MocapDataset(root=root)
    test_dset = MocapDataset(root=root, test=True)
    train_loader = DataLoader(train_dset, batch_size=batch_size)
    test_loader = DataLoader(test_dset, batch_size=batch_size)
    return train_loader, test_loader


if __name__ == "__main__":
    import random
    mocap_dataset = MocapDataset()

    idx = random.randint(0, len(mocap_dataset))
    j,c = mocap_dataset[idx]

    print('Joint shape: ', j.shape)
    print('Joint mean: ', j.mean)
    print('Joint min: {}, max: {} '.format(j.min(), j.max()))
    print('Control shape: ', c.shape)
    print('Contorl mean: ', c.mean())
    print('Contorl min: {}, max: {} '.format(c.min(), c.max()))
