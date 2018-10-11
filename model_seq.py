import tensorflow as tf
import numpy as np
from base_model import BaseModel
  

class Multi_Label_Class(BaseModel):
    def build(self):
        """ Build all tensorflow model by called each method...... """
        self.build_encoder()
        self.build_rnn()
        if self.is_train: # eval
            self.build_optimizer()
            self.build_summary()


    def build_encoder(self):
        """ Build the VGG16 net. """
        print("Building CNN model (SqS encoder)..........")
        config = self.config

        # Input 
        self.images = tf.placeholder(
                    dtype = tf.float32,
                    shape = self.image_shape) #image_shape in base_model.py

        '''Building LSTM cell with Dropout..........'''
        def get_a_cell(num=config.num_lstm_units, 
                       drop_rate=config.lstm_drop_rate):
            lstm = tf.nn.rnn_cell.LSTMCell(
                             num,
                             initializer = self.nn.fc_kernel_initializer)
            if self.is_train:
                lstm = tf.nn.rnn_cell.DropoutWrapper(
                      lstm,
                      input_keep_prob = 1.0 - drop_rate,
                      output_keep_prob = 1.0 - drop_rate,
                      state_keep_prob = 1.0 - drop_rate)
            return lstm


        '''Initializing input data using the mean context...'''
        with tf.variable_scope('encoder', initializer=tf.orthogonal_initializer()):
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                              [get_a_cell() for layer in range(4)]
                                                      )
            self.initial_state = encoder_cell.zero_state(config.batch_size, tf.float32)
            # 透過dynamic_rnn對cell展開時間維度
            self.encoder_outputs, self.encoder_state  = tf.nn.dynamic_rnn(
                                                              encoder_cell, 
                                                              self.images,                                                    
                                                              initial_state=self.initial_state
                                                                          )

    



#=========================================================
    def build_rnn(self):
        ''' Build the RNN................... '''
        print("Building the RNN Model..........")
        config = self.config


        # Setup the placeholders
        contexts = self.encoder_outputs # come from Encoder output
        self.labels = tf.placeholder(
                      dtype = tf.float32,
                      shape = [config.batch_size, 
                               1, 
                               config.label_index_length])


        '''Building LSTM cell with Dropout..........'''
        lstm = tf.nn.rnn_cell.LSTMCell(
                     config.num_lstm_units,
                     initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                      lstm,
                      input_keep_prob = 1.0-config.lstm_drop_rate,
                      output_keep_prob = 1.0-config.lstm_drop_rate,
                      state_keep_prob = 1.0-config.lstm_drop_rate)


        '''Initializing input data using the mean context...'''
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.encoder_outputs, axis = 1) # take mean of CNN output
            initial_memory, initial_output = self.initialize(context_mean) #Call initialize()
            #initial_state = initial_memory, initial_output



        ''' Prepare to run model...................'''
        num_steps = config.max_class_label_length
        probability = tf.zeros([config.batch_size, config.label_index_length], tf.float32)
        hard_label = tf.zeros([config.batch_size, config.label_index_length], tf.float32)
        last_memory = initial_memory # C after initialized 
        last_output = initial_output
        last_state = last_memory, last_output

        result_max_idx = []
        result_max_value = []
        if self.is_train:
            alphas = [] # Parameters in attention operation
            cross_entropies = []


        ''' LSTM predict with time step:  max_class_label_length'''
        for _ in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output) # After Softmax 
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                if self.is_train:
                    alphas.append(tf.reshape(alpha, [-1]))

            with tf.variable_scope("lstm"):
                current_input = tf.concat([context,  #attention input 
                                           probability,     
                                           hard_label], 1)
                output, state = lstm(current_input, last_state) # state = (C, h)
                memory, _ = state


            '''Decode the expanded output of LSTM into a word'''
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             probability,
                                             hard_label],
                                             axis = 1)
                '''decode(): Last nn layers for predict'''                              
                logits = self.decode(expanded_output)
                label_reshape = tf.reshape(self.labels,  logits.get_shape())
                probs = tf.nn.sigmoid(logits)  # Become next input


            '''Generator Hard lable'''
            compare_probs = tf.subtract(probs, hard_label)
            max_value = tf.reduce_max(compare_probs, axis=1)
            max_id = tf.argmax(compare_probs, axis=1)
            hard_label = tf.add(hard_label, 
                            tf.cast(
                                tf.equal(compare_probs, 
                                    tf.expand_dims(max_value, 1)), tf.float32))
            if not self.is_train:
                result_max_value.append(max_value)
                result_max_idx.append(max_id)



            """ Compute the loss for this step, if necessary. """
            if self.is_train:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                labels = label_reshape,
                                logits = logits)
                cross_entropies.append(cross_entropy) #loss of each step

                # next step input
                probability = probs
                last_output = output
                last_memory = memory
                last_state = state
            tf.get_variable_scope().reuse_variables()
        '''End: for loop'''
        

        if not self.is_train:
            self.final_prob_predict = tf.identity(probs, name='final_prob_predict')
            self.final_result_max_idx = tf.stack(result_max_idx)
            self.final_result_max_value = tf.stack(result_max_value)


        # Compute the final loss in Training Process
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_mean(cross_entropies)

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss +  reg_loss

            #predictions_correct = tf.stack(predictions_correct, axis = 1)
            #accuracy = tf.reduce_sum(predictions_correct)

            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.contexts = contexts
        print("RNN built...........................")
        ''' End: build_rnn() '''




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
        reshaped_contexts = tf.reshape(contexts, [-1, config.num_lstm_units])
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
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, config.time_step, 1])
        temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
        temp = temp1 + temp2
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'fc_2')
        logits = tf.reshape(logits, [-1, config.time_step])
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
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            #tf.summary.scalar("accuracy", self.accuracy)
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
