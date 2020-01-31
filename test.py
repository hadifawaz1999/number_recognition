import numpy as np
from DATA import load_data_mnnist
import matplotlib.pyplot as plt

path="/home/hadi/data_sets/keras/mnist/"
xtrain=np.load(path+"x_train.npy")
print(xtrain.shape)
plt.imshow(xtrain[4],cmap='gray')
plt.show()