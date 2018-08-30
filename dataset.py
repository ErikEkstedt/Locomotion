import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# TODO
# Easy validation split


class MocapDataset(Dataset):
    def __init__(self, path='mocap/data/edinburgh_locomotion_train.npz'):
        self.data = torch.from_numpy(np.load(path)['clips'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions = self.data[idx][:, :-3]
        ctrl = self.data[idx][:, -3:]
        return {'j_pos': joint_positions, 'ctrl':ctrl}


class MocapDataLoader(DataLoader):
    def __init__(self, dset, batch_size, seq_len=1, overlap_len=0, *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            jpos = batch['j_pos']
            ctrl = batch['ctrl']
            batch_size, n_frames, n_joints = jpos.shape

            reset = True
            for i in range(self.overlap_len, n_frames, self.seq_len):
                j = jpos[:, i, :]
                c = ctrl[:, i, :]
                yield j, c, reset
                reset = False


def dataset(batch_size=32,
            seq_len=240,
            overlap_len=0,
            path='mocap/data/edinburgh_locomotion_train.npz',
            *args, **kwargs):
    dset = MocapDataset()
    dloader = MocapDataLoader(dset, batch_size=32, seq_len=60)
    return dloader

