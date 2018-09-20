import os
import math
import numpy as np
import pandas as pd
from config import Config
from tqdm import tqdm

class DataSet(object):
    def __init__(self, config):
        self.config = config

    def train_data(self):
        chunksize = self.config.batch_size * 500
        X_train = pd.read_csv(self.config.X_train_data, chunksize=chunksize)
        Y_train = pd.read_csv(self.config.Y_train_data, chunksize=chunksize)
        '''Data size: [batc_size, h, w, rgb]'''

        for X_chunk, Y_chunk in(X_train, Y_train):
            X_batch = np.zeros((self.config.batch_size, 
                                self.config.data_step, 
                                data_size, 
                                1, ))
            Y_batch =  


            for  







        return 0
    
    def eval_data(self):
        evaldata=0
        return evaldata
        
    def test_data(self):
        testdata=0
        return testdata
