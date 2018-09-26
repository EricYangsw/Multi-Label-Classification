import tensorflow as tf
import numpy as np
from base_model import BaseModel
  

class Multi_Label_Class(BaseModel):
    def build(self):
        """ Build all tensorflow model by called eachmethod...... """
        self.build_vgg16()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()


    def build_vgg16(self):
        """ Build the VGG16 net. """
        print("Building CNN model (vgg16)..........")
        config = self.config

        # Input 
        images = tf.placeholder(
                    dtype = tf.float32,
                    shape = self.image_shape) #image_shape in base_model.py
        '''hight_out = (hight_in - hight_f + 2*padding)/stride  +1
           width_out = (width_in - width_f + 2*padding)/stride  +1 
        '''
        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')        
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2') 
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')     

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 13, 512])
        self.conv_feats = reshaped_conv5_3_feats # CNN output (into RNN)

        self.num_ctx = 13
        self.dim_ctx = 512
        self.images = images #?




#======================================================
    def build_rnn(self):
        """ Build the RNN................... """
        print("Building the RNN Model..........")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats # come from CNN output
            self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, 1, config.label_index_length])
        else:
            contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder( #?????
                dtype = tf.int32,
                shape = [config.batch_size])



        """Building LSTM cell with Dropout.........."""
        lstm = tf.nn.rnn_cell.LSTMCell(
                     config.num_lstm_units,
                     initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                      lstm,
                      input_keep_prob = 1.0-config.lstm_drop_rate,
                      output_keep_prob = 1.0-config.lstm_drop_rate,
                      state_keep_prob = 1.0-config.lstm_drop_rate)



        """Initializing input data using the mean context"""
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1) # take mean of CNN output
            initial_memory, initial_output = self.initialize(context_mean) #Call initialize()
            #initial_state = initial_memory, initial_output


        """ Prepare to run model..................."""
        #predictions = []
        step_max_list = []
        
        if self.is_train:
            alphas = [] # Parameters in attention operation
            cross_entropies = []
            #predictions_correct = []
            num_steps = config.max_class_label_length

            probability = tf.zeros([config.batch_size, config.label_index_length], tf.float32)
            hard_label = tf.zeros([config.batch_size, config.label_index_length], tf.float32)
            last_memory = initial_memory # C after initialized 
            last_output = initial_output
        else:
            num_steps = 1
        last_state = last_memory, last_output


        '''initializing variabel for pick label'''
        pick_hard_label = []


        """ LSTM predict with time step:  max_class_label_length"""
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output) # After Softmax
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                          # tf.expend_dim(): Inserts a dim of 1 into tensor

                if self.is_train:
                    alphas.append(tf.reshape(alpha, [-1]))

            with tf.variable_scope("lstm"):
                current_input = tf.concat([context,  #attention input 
                                           probability,     # last_oupput
                                           hard_label], 1)
                output, state = lstm(current_input, last_state) # state = (C, h)
                memory, _ = state


            """Decode the expanded output of LSTM into a word"""
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             probability,
                                             hard_label],
                                             axis = 1)
                """decode(): Last nn layers for predict"""                              
                logits = self.decode(expanded_output)
                label_reshape = tf.reshape(self.labels,  logits.get_shape())
                probs = tf.nn.sigmoid(logits)     # Become next input



                """Generator Hard lable"""
                pre_max_label = tf.argmax(probs)
                if idx == 0:
                    pick_hard_label.append(pre_max_label)
                else:
                    step_max = tf.argmax(probs, axis=1)
                    step_max_list.append(step_max)
                    step_max_tensor = tf.stack(step_max_list, axis=1)
                    
                    compare_max_list = []
                    for i in range(idx):
                        compare_max_list.append(step_max)
                    compate_max_tensor = tf.stack(compare_max_list, axis=1)

                    true_false_tensor = tf.reduce_sum(
                                            tf.cast(
                                                tf.equal(compate_max_tensor, step_max_tensor), 
                                            dtype=tf.int32), 
                                        axis=1)

                    pred = []
                    for i in range(config.batch_size):
                        new_step_max = tf.cond( tf.equal(true_false_tensor[i], 0), 
                                           true_fn=lambda: self.cond_true_fun(step_max[i]), # True
                                           false_fn=lambda: self.cond_false_fun(probs[i], step_max_tensor[i])) # False
                        pred.append(tf.one_hot(new_step_max, 
                                               depth=config.label_index_length, 
                                               dtype=tf.float32))
                    hard_label = tf.add(hard_label, tf.stack(pred, axis=0))
                #predictions.append(prediction)


            """ Compute the loss for this step, if necessary. """
            if self.is_train:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = label_reshape,
                    logits = logits)
                cross_entropies.append(cross_entropy)


                #ground_truth = tf.cast(label_reshape, tf.int64)
                #prediction_correct = tf.where(
                #    tf.equal(prediction, ground_truth),
                #    tf.cast(tf.zeros_like(prediction), tf.float32))
                #predictions_correct.append(prediction_correct)


                # prepare to next step
                probability = probs
                last_output = output
                last_memory = memory
                last_state = state
            tf.get_variable_scope().reuse_variables()
        """End: for loop"""


        # Compute the final loss
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies)


            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = (config.attention_loss_factor * 
                              tf.nn.l2_loss(diffs) /
                              (config.batch_size * self.num_ctx))

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss + attention_loss + reg_loss

            #predictions_correct = tf.stack(predictions_correct, axis = 1)
            #accuracy = tf.reduce_sum(predictions_correct)



        self.contexts = contexts
        if self.is_train:
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            #self.accuracy = accuracy
            self.attentions = attentions
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs
        print("RNN built...........................")
        ''' End: build_rnn() '''



    """ function for tf.cond()"""
    def cond_true_fun(self, step_max):
        return step_max

    def cond_false_fun(self, probs, step_max_tensor): 
        config = self.config
        step_max_tensor = tf.cast(step_max_tensor, tf.int32)
        probs = tf.subtract(
                    probs, tf.reduce_sum(
                                tf.one_hot(
                                step_max_tensor, 
                                depth=config.label_index_length, 
                                dtype=tf.float32), 
                            axis=0))
        new_step_max = tf.argmax(probs, axis=0, output_type=tf.int64)
        return new_step_max    



    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        # use 2 fc layers to initialize
        temp1 = self.nn.dense(context_mean,
                                units = config.dim_initalize_layer, 
                                activation = tf.tanh,
                                name = 'fc_a1')
        temp1 = self.nn.dropout(temp1)
        memory = self.nn.dense(temp1,
                                units = config.num_lstm_units, 
                                activation = None,
                                name = 'fc_a2')

        temp2 = self.nn.dense(context_mean,
                                units = config.dim_initalize_layer,
                                activation = tf.tanh,
                                name = 'fc_b1')
        temp2 = self.nn.dropout(temp2)
        output = self.nn.dense(temp2,
                                units = config.num_lstm_units,
                                activation = None,
                                name = 'fc_b2')
        return memory, output



    def attend(self, contexts, output):
        """ Attention Mechanism....."""
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        # use 2 fc layers to attend
        temp1 = self.nn.dense(reshaped_contexts,
                                units = config.dim_attend_layer,
                                activation = tf.tanh,
                                name = 'fc_1a')
        temp2 = self.nn.dense(output,
                                units = config.dim_attend_layer,
                                activation = tf.tanh,
                                name = 'fc_1b')
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
        temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
        temp = temp1 + temp2
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'fc_2')
        logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha



    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word...."""
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        # use 2 fc layers to decode
        temp = self.nn.dense(expanded_output,
                                units = config.dim_decode_layer,
                                activation = tf.tanh,
                                name = 'fc_1')
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                                units = config.label_index_length,
                                activation = None,
                                name = 'fc_2')
        return logits



    def build_optimizer(self):
        """ opt_op : 
            Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None


        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )
            opt_op = tf.contrib.layers.optimize_loss(
                        loss = self.total_loss,
                        global_step = self.global_step,
                        learning_rate = learning_rate,
                        optimizer = optimizer,
                        clip_gradients = config.clip_gradients,
                        learning_rate_decay_fn = learning_rate_decay_fn)
        self.opt_op = opt_op



    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)
        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            #tf.summary.scalar("accuracy", self.accuracy)
        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)
        self.summary = tf.summary.merge_all()



    def variable_summary(self, var):
        """ Build the summary for a variable...."""
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
