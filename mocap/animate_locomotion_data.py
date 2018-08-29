import numpy as np

from AnimationPlotLines import animation_plot

data = np.load('data/edinburgh_locomotion_train.npz')

clips = data['clips']
clips = np.swapaxes(clips, 1, 2)

index = 0
seq1 = clips[index:index+1]
seq2 = clips[index+1:index+2]

animation_plot([seq1, seq2], interval=15.15)
