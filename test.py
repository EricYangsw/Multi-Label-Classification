import os 
import numpy as np
import tensorflow as tf

save_path = './save_models/0.npy'


data = np.array([1,2,3])
np.save(save_path, data)