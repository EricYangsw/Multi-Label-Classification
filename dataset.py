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
        self.train_ratio = self.config.train_ratio
        self.num_batches = int((4000-9)*34/config.batch_size)
        self.num_eval_batches = int((4000-9)*8/config.batch_size)



    def read_file(self, path='./data/devide_data/'):
        self.path_input = path + 'y'
        self.path_label = path + 'x'
        filelist_input = os.listdir(self.path_input)
        filelist_label = os.listdir(self.path_label)
        assert len(filelist_input) == len(filelist_label) #False to tigger

        file_count = len(filelist_input)
        train_file_count = int(file_count*self.train_ratio)
        return file_count, train_file_count



    def train_data(self):
        file_count, train_file_count = self.read_file()
        '''Data size: [batc_size, h, w, rgb]'''
        for no_file in range(train_file_count):
            batch_input = np.zeros((self.config.batch_size, 
                                self.config.time_step, 
                                self.config.fearute_size,
                                1))
            batch_label =  np.zeros((self.config.batch_size,
                                 1, #time step
                                 self.config.label_index_length))

            data_input = pd.read_csv(self.path_input + '/' + 
                                     'y_data_' +
                                     str(no_file)+'.csv', 
                                     index_col=0).values
            data_label = pd.read_csv(self.path_label+'/' + 
                                     'x_data_' +
                                     str(no_file)+'.csv', 
                                     index_col=0).values                          
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
        file_count, train_file_count = self.read_file()
        '''Data size: [batc_size, h, w, rgb]'''
        for no_file in range(train_file_count, file_count):
            batch_input = np.zeros((self.config.batch_size, 
                                self.config.time_step, 
                                self.config.fearute_size,
                                1))
            batch_label =  np.zeros((self.config.batch_size,
                                 1,  # time step
                                 self.config.label_index_length))

            data_input = pd.read_csv(self.path_input + '/' +
                                     'y_data_' + 
                                     str(no_file) + 
                                     '.csv', index_col=0).values
            data_label = pd.read_csv(self.path_label + '/' +
                                     'x_data_' +
                                     str(no_file) + 
                                     '.csv', index_col=0).values
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


