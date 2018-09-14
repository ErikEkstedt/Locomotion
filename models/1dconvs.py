import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



x = torch.ones((1,40)).unsqueeze(0)
layer = nn.Conv1d(in_channels=1,
                  out_channels=3,
                  stride=1,
                  kernel_size=5,
                  padding=True)
out = layer(x)
out.shape


