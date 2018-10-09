import os
import pickle
import copy
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn import NN
from dataset import DataSet
 

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_shape = [config.batch_size, 
                            config.time_step, 
                            config.fearute_size, 
                            1] # input shape
        self.nn = NN(config) # Base cnn unit 
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build() #Run building method
    def build(self):
        """Prepare to be overrided in child class"""
        raise NotImplementedError()



    def load(self, sess, model_file=None):
        """ Load the model........."""
        config = self.config
        
        """Make save_path"""
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")
        """Load model by np.load()"""
        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item() 
        #"np.load": Load arrays or pickled objects from .npy, .npz or pickled files.
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded....." %count)
  


    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path, encoding = 'latin1').item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                print('Variabel name: ', op_name) # Make sure variable name
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name, trainable=False)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)



    def train(self, sess):
        print("Training the model...........")
        config = self.config
        loss_train_data = []

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        
        for epoch in tqdm(list(range(config.num_epochs)), desc='epoch'):
            print("Training data generator in {} Epoch...".format(epoch))
            make_data = DataSet(config)
            train_data = make_data.train_data()
            for ba in tqdm(list(range(make_data.num_batches)), desc='batch'):
                try:
                    batch = train_data.__next__()
                except:
                    loss_train_data.append(_)
                    continue
                images, labels = batch
                feed_dict = {self.images: images, # in model,py
                             self.labels: labels} # in model,py
                _, summary, cross_entropy_loss, global_step  = sess.run([
                                                             self.opt_op, #in model.build_optimizer()
                                                             self.summary,#in model.build_summary()
                                                             self.cross_entropy_loss,
                                                             self.global_step],
                                                             feed_dict=feed_dict)
                print('gobal_step', global_step)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                if (global_step + 1) % config.show_loss == 0:
                    print('epoch: {}, batch: {}, Loss: {}'.format(epoch, ba, cross_entropy_loss))
                train_writer.add_summary(summary, global_step)
        train_writer.close()
        print("Training complete.......")
        print("loss_train_data: ", loss_train_data)




    def evals(self, sess):
        print("Evaluating the model.......")
        config = self.config
        loss_eval_data = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        make_data = DataSet(config)
        eval_data = make_data.eval_data()

        count = 0
        total = 0
        for i in tqdm(list(range(make_data.num_eval_batches)), desc='batch'):
            try:
                batch = eval_data.__next__()
            except:
                loss_eval_data.append(i)
                continue
            images, labels = batch
            feed_dict = {self.images: images, # in model,py
                            self.labels: labels} # in model,py
            final_prob_predict, final_result_max_idx, final_result_max_value = sess.run(
                                            [self.final_prob_predict, 
                                            self.final_result_max_idx,
                                            self.final_result_max_value], 
                                            feed_dict=feed_dict)

            
            
            label_reshape = np.reshape(labels,  final_prob_predict.shape)

            c, t = self.error(final_prob_predict, label_reshape, i)
            count += c
            total += t
            


            #np.savetxt("./val_results/final_prob_predict_" + str(i) + ".csv", final_prob_predict, delimiter=',')
            #np.savetxt("./val_results/labels_" + str(i) + ".csv", label_reshape, delimiter=',')
            np.savetxt("./val_results/final_result_max_idx_" + str(i) + ".csv", final_result_max_idx, delimiter=',')
            np.savetxt("./val_results/final_result_max_value_" + str(i) + ".csv",final_result_max_value, delimiter=',')
        self.err = count / total
        print('Total err: ', self.err)
        print("Evaluation complete........")
        print("Loss batch eval data: ", loss_eval_data)
    

    def error(self, pred, target, i):
        pred = np.array(pred >= 0.5).astype(int)
        result = np.abs(pred - target)
        count = np.sum(result)
        total = result.size
        np.savetxt("./val_results/result_" + str(i) + ".csv", result, delimiter=',')
        return count, total


    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved......")
