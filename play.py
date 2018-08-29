import numpy as np

from mocap.AnimationPlotLines import animation_plot

train_data = np.load('mocap/data/edinburgh_locomotion_train.npz')
test_data = np.load('mocap/data/edinburgh_locomotion_test.npz')

clips = train_data['clips']
clips = np.swapaxes(clips, 1, 2)

index = 0
seq1 = clips[index:index+1]
seq2 = clips[index+1:index+2]

animation_plot([seq1, seq2], interval=15.15)


