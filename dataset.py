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
        self.num_batches = int((4000-9)*34/config.batch_size)
        self.num_eval_batches = int((4000-9)*8/config.batch_size)



    def train_data(self):   
        #chunksize = self.config.batch_size * 500 # dataframe of iter
        #X_train = pd.read_csv(self.config.Y_train_data, chunksize=chunksize)
        #Y_train = pd.read_csv(self.config.X_train_data, chunksize=chunksize)

        path_input = './data/devide_data/y'
        path_label = './data/devide_data/x'
        filelist_input = os.listdir(path_input)
        filelist_label = os.listdir(path_label)
        assert len(filelist_input) == len(filelist_label) #False to tigger

        file_count = len(filelist_input)

        '''Data size: [batc_size, h, w, rgb]'''
        for no_file in range(34):
            batch_input = np.zeros((self.config.batch_size, 
                                self.config.time_step, 
                                self.config.fearute_size,
                                1))
            batch_label =  np.zeros((self.config.batch_size,
                                 1,  # time step
                                 self.config.label_index_length))

            data_input = pd.read_csv(path_input+'/'+'y_data_'+str(no_file)+'.csv', index_col=0).values
            data_label = pd.read_csv(path_label+'/'+'x_data_'+str(no_file)+'.csv', index_col=0).values
            assert data_input.shape[0] == data_label.shape[0]

            data_count = data_input.shape[0]
            batch = 0
            for i in range(self.config.time_step, data_count):
                batch_input[batch, :, :, 0] = data_input[i-self.config.time_step:i, :]
                batch_label[batch, :] = data_label[i, :]
                batch += 1
                if batch == self.config.batch_size:
                    batch = 0
                    yield (batch_input, batch_label)
                     



    
    def eval_data(self):
        #chunksize = self.config.batch_size * 500 # dataframe of iter
        #X_train = pd.read_csv(self.config.Y_train_data, chunksize=chunksize)
        #Y_train = pd.read_csv(self.config.X_train_data, chunksize=chunksize)

        path_input = './data/devide_data/y'
        path_label = './data/devide_data/x'
        filelist_input = os.listdir(path_input)
        filelist_label = os.listdir(path_label)
        assert len(filelist_input) == len(filelist_label) #False to tigger

        file_count = len(filelist_input)
        batch = 0
        '''Data size: [batc_size, h, w, rgb]'''
        for no_file in range(34, file_count):
            batch_input = np.zeros((self.config.batch_size, 
                                self.config.time_step, 
                                self.config.fearute_size,
                                1))
            batch_label =  np.zeros((self.config.batch_size,
                                 1,  # time step
                                 self.config.label_index_length))

            data_input = pd.read_csv(path_input+'/'+'y_data_'+str(no_file)+'.csv', index_col=0).values
            data_label = pd.read_csv(path_label+'/'+'x_data_'+str(no_file)+'.csv', index_col=0).values
            assert data_input.shape[0] == data_label.shape[0]

            data_count = data_input.shape[0]
            batch = 0
            for i in range(self.config.time_step, data_count):
                batch_input[batch, :, :, 0] = data_input[i-self.config.time_step:i, :]
                batch_label[batch, :] = data_label[i, :]
                batch += 1
                if batch == self.config.batch_size:
                    batch = 0
                    yield (batch_input, batch_label)
