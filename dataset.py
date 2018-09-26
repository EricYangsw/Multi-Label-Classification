import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from config import Config
from tqdm import tqdm
       
  
class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.num_batches = 0

    def train_data(self):   
        chunksize = self.config.batch_size * 500 # dataframe of iter
        X_train = pd.read_csv(self.config.Y_train_data, chunksize=chunksize)
        Y_train = pd.read_csv(self.config.X_train_data, chunksize=chunksize)


        '''Data size: [batc_size, h, w, rgb]'''
        for X_chunk, Y_chunk in(X_train, Y_train):
            '''Deal with each chunk.........'''
            X_batch = np.zeros((self.config.batch_size, 
                                self.config.time_step, 
                                self.config.fearute_size,
                                1))
            Y_batch =  np.zeros((self.config.batch_size,
                                 1,  # time step
                                 self.config.fearute_size))

            data_count = X_batch.shape[0]
            batch = 0

            for i in range(self.config.time_step, data_count):
                X_batch[batch, :, :, 0] = X_chunk[i-self.config.time_step:i, :]
                Y_batch[batch, :] = Y_chunk[i, :]
                batch += 1
                if batch == self.config.batch_size:
                    batch = 0
                    self.num_batches += 1
                    yield (X_batch, Y_batch)
                     



    
    def eval_data(self):
        evaldata=0
        return evaldata
        
    def test_data(self):
        testdata=0
        return testdata
