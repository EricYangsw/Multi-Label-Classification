class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.num_lstm_units = 512  
        self.dim_initalize_layer = 512
        self.dim_attend_layer = 512
        self.dim_decode_layer = 512


        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.5
        self.attention_loss_factor = 0.01
        

        # Data size
        self.time_step = 70
        self.max_class_label_length = 146 # the number of "1" label in each data
        self.label_index_length = 223
        self.fearute_size = 138
        self.train_ratio = 0.8


        # about the optimization
        self.num_epochs = 200
        self.batch_size = 32
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6


        # about the saver
        self.save_period = 1000
        self.show_loss = 20
        self.save_dir = './save_models/'
        self.summary_dir = './summary_8k/'


        # about the training
        self.X_train_data = './x_input.csv'
        self.Y_train_data = './y_label.csv'

  
        # about the evaluation
        self.eval_result_dir = './val_results/'
        self.eval_result_file = './val_results/results.json'
        self.save_eval_result_as_image = False


        # about the testing
        self.test_result_dir = './test_results/'
        self.test_result_file = './test_results/results.csv'
     
     