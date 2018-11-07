import numpy as np
import tensorflow as tf
from utils import define_scope
from utils import make_train_op
from utils import normalized_columns_initializer

class Gym_ac_model(object):
    def __init__(self,
                 num_actions,
                 ob_dim,
                 lstm = True,
                 trainable = True,
                 scope = 'global',
                 frame_per_obs = 4,
                 lost_function = 'ppo',
                 ):
        self.num_actions = num_actions
        self.ob_dim = ob_dim
        self.scope = scope
        self.frame_per_obs = frame_per_obs
        self.lost_function = lost_function
        
        ## create model
        self._place_holders()
        ## lost function support a2c and ppo
        if lost_function == 'a2c':
            self.embedding = self._frame2embed_conv(self.cur_obs_ph)
            self.rnn_out,self.rnn_state = self._lstm(self.rnn_init,self.embedding)
            self.policy = self._policy(self.rnn_out)
            self.value = self._value(self.rnn_out)
            self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.local_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            if trainable:        
                self._loss()
                self._apply_gradient()
        elif lost_function == 'ppo':            
            with tf.variable_scope('current',reuse=False):
                self.embedding = self._frame2embed_conv(self.cur_obs_ph)
                self.rnn_out,self.rnn_state = self._lstm(self.rnn_init,self.embedding)
                self.policy = self._policy(self.rnn_out)
                self.value = self._value(self.rnn_out)
            self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/current')
            
            if trainable:
                with tf.variable_scope('prev',reuse=False):
                    self.embedding_pre = self._frame2embed_conv(self.cur_obs_ph)
                    self.rnn_out_pre,_ = self._lstm(self.rnn_init,self.embedding_pre)
                    self.policy_pre = self._policy(self.rnn_out_pre)
                    self.value_pre = self._value(self.rnn_out_pre)
                                    
                self.local_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/current')
                self.local_pre_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/pre')            

                self.update_pre_variabls=self.update_variables(self.local_vars,self.local_pre_vars)
                self._ppo_loss()
                self._apply_gradient()
            
        ## if not global, add function to copy variables from global            
        if self.scope != 'global':
            self.update_local_variables=self.update_variables(self.global_vars,self.local_vars)
                            


    def _place_holders(self):
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name='learning_rate')
        if self.lost_function == 'ppo':
            self.cliprange = tf.placeholder(dtype=tf.float32,shape=[],name='cliprange')

        self.cur_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None]+list(self.ob_dim), name='cur_obs')
        ## TD target
        self.advs_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="advs")
        self.target_values_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="values")
        ## actions
        self.actions_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        ## lstm
        self.rnn_init = tf.placeholder(tf.float32, (None, 512), name='rnn_init')

        
    
    def _frame2embed_conv(self,obs):
        
        with tf.variable_scope("conv", reuse=False):

            ## align with open ai universal starter implementation
            out = obs        
            out = tf.contrib.layers.conv2d(
                out, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
            out = tf.contrib.layers.conv2d(
                out, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
            
            # Fully connected layer
            embedding = tf.contrib.layers.fully_connected(
                            inputs=tf.contrib.layers.flatten(out),
                            num_outputs=256,
                            scope="fc1")
            return embedding

    def _lstm(self,rnn_init,embedding):
        with tf.variable_scope('lstm'):

            with tf.variable_scope('state_conversion'):                
                h_in,c_in=tf.split(rnn_init,[256,256],1)
                init_state = tf.contrib.rnn.LSTMStateTuple(h_in, c_in)
            
            
        #Recurrent network for temporal dependencies
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
        
        rnn_in = tf.expand_dims(embedding, [1])
            
        rnn_out, rnn_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=init_state)
        
        ## reduce rnn_out dimension
        rnn_out = tf.squeeze(rnn_out,axis=1)
        rnn_state = tf.concat([rnn_state[0], rnn_state[1]], 1)
        return(rnn_out,rnn_state)
                
    def _policy(self,rnn_out):
        policy = tf.layers.dense(inputs = rnn_out,
                                  units = self.num_actions,
                                  activation=tf.nn.softmax,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  name='policy')
        ##avoid nan
        policy = tf.clip_by_value(policy, 1e-20, 1.0)
        return policy
    def _value(self,rnn_out):
        value = tf.layers.dense(inputs = rnn_out,
                                  units = 1,
                                  activation=None,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=1),
                                  name='value')
        value = tf.squeeze(value, squeeze_dims=[1], name="value_squeeze")
        return value
    
    def _loss(self):
        
        ## policy loss
        with tf.variable_scope("loss"):
            
            ##value loss
            self.value_loss = tf.reduce_sum(tf.nn.l2_loss(self.target_values_ph-self.value),name='value_loss')

            use_log_epsilon = False
            if use_log_epsilon:
                self.log_epsilon = 1e-6            
                log_policy = tf.log(tf.maximum(self.policy,self.log_epsilon))
            else:
                log_policy = tf.log(self.policy)
            # We add entropy to the loss to encourage exploration 
            self.entropy_mean = - tf.reduce_sum(tf.reduce_sum(self.policy*log_policy,[1]), name="entropy_sum")
            ## policy loss
            ac_onehot = tf.one_hot(indices=self.actions_ph, depth=self.num_actions)
            self.policy_loss = - tf.reduce_sum(tf.reduce_sum(log_policy * ac_onehot, [1])*self.advs_ph)
                
            
            self.total_loss = 0.5*self.value_loss+self.policy_loss - self.entropy_mean * 0.01

    def _ppo_loss(self):
        
        ## policy loss
        with tf.variable_scope("loss"):
            value_clip = self.value_pre + tf.clip_by_value(self.value - self.value_pre, - self.cliprange, self.cliprange)
            vf_losses1 = tf.square(self.value - self.target_values_ph)
            vf_losses2 = tf.square(value_clip - self.target_values_ph)
            self.value_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2),name='value_loss')

            log_policy = tf.log(self.policy)
            self.entropy_mean = - tf.reduce_mean(tf.reduce_sum(self.policy*log_policy,[1]), name="entropy_sum")
            ## policy loss
            ac_onehot = tf.one_hot(indices=self.actions_ph, depth=self.num_actions)
            pre_neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=self.policy_pre,labels=ac_onehot)
            neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=self.policy,labels=ac_onehot)
            ratio = tf.exp(pre_neglogpac - neglogpac)
            pg_losses1 = -self.advs_ph * ratio
            pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.policy_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            
            self.total_loss = self.value_loss+self.policy_loss - self.entropy_mean * 0.01

    def _apply_gradient(self):
        grads = tf.gradients(self.total_loss,self.local_vars)
        grads, self.orig_norm = tf.clip_by_global_norm(grads, 0.5)
        self.apply_gradient = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(grads,self.global_vars))

        
        

    def update_variables(self,from_vars,to_vars):
        ## update local variables with global variables
        self._update_variables = []
        for from_var,to_var in zip(from_vars,to_vars):
            self._update_variables.append(to_var.assign(from_var))
        return self._update_variables
    
        
        


