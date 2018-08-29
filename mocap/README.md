# The Edinburgh Locomotion MOCAP Database #

This dataset contains 1855 different sequences of human motion captured with a MOCAP system 
at the School of Informatics, The University of Edinburgh. The data was captured on two separate
occasions during which all actions were performed by the same male actor. The datasets contains 
mainly human locomotion data, including walking, jogging and running.

We partition the data into 1835 examples in the training set 'data/edinburgh_locomotion_train.npz' 
as well as 20 additional examples in a separate test set 'data/edinburgh_locomotion_test.npz'.

### Data description ###

We represent a motion sequence at 60Hz through the (x, y, z) coordinates of 21 joint positions at each
time step. In addition, we provide 3 scalar control signals (forward, sideways and rotational velocity) 
with each frame of the sequence. Thus, a single frame is a 66 dimensional vector. Moreover,
each sequence in the provided dataset consists of 240 frames (4 seconds) and is padded with the first and
last frame where necessary.

Each value corresponds to the following marker/control signal:

~~~~
0-2: Hip
3-5: Left Hip Joint
6-8: Left Knee
9-11: Left Heel
12-14: Left Toe
15-17: Right Hip Joint
18-20: Right Knee
21-23: Right Heel
24-26: Right Toe
27-29: Lower Back
30-32: Spine
33-35: Neck
36-38: Head
39-41: Left Shoulder
42-44: Left Elbow
45-47: Left Hand
48-50: Left Hand Index
51-53: Right Shoulder
54-56: Right Elbow
57-59: Right Hand
60-62: Right Hand Index
63: Forward Velocity
64: Sideways Velocity
65: Rotational Velocity
~~~~


### Citing this data set ###

* If you use this dataset in your research, please consider citing the following paper:

~~~~
@inproceedings{ikhansul17-vaelstm,
 author={Habibie, Ikhansul and Holden, Daniel and Schwarz, Jonathan and Yearsley, Joe and Komura, Taku},
 booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
 title = {A Recurrent Variational Autoencoder for Human Motion Synthesis},
 year = {2017}
}
~~~~

### Python ###
* You can explore this dataset using the code in `animate_locomotion_data.py` which loads the training set and shows two motion sequences:

~~~
import numpy as np

from AnimationPlotLines import animation_plot

data = np.load('data/edinburgh_locomotion_train.npz')

clips = data['clips']
clips = np.swapaxes(clips, 1, 2)

index = 0
seq1 = clips[index:index+1]
seq2 = clips[index+1:index+2]

animation_plot([seq1, seq2], interval=15.15)
~~~