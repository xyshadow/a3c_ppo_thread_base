import sys
import os
import itertools
import numpy as np
import tensorflow as tf
import time

from gym_atari_ac_model import Gym_ac_model
from gym_atari_env import Gym_atari_env
from utils import discount
from utils import timetest
from multiprocessing import Process, Queue, Value, Event
from datetime import datetime
import time
from threading import Thread
from actor import Actor

## server is a combination of predictor and trainer, which contains its only model
class Server(object):
    def __init__(self,
                 name,
                 env_name,
                 max_train_step,
                 num_actors,
                 horizon,                 
                 mbatch,
                 num_epoch,
                 log_dir,
                 num_actions,
                 ob_dim,
                 frame_per_state=1,
                 lstm=True,
                 lstm_state=256,
                 lr = 2.5e-4,
                 cliprange = 0.1,
                 train_step = Value('i',0),
                 predict_counter = Value('i',0),
                 last_reward = Value('i',0),
                 last_length = Value('i',0),                 
                 env_class=Gym_atari_env):
        
        self.name = name
        self.env_name = env_name
        self.max_train_step = max_train_step
        self.num_actors = num_actors
        self.horizon = horizon
        self.mbatch = mbatch
        self.num_epoch = num_epoch
        self.log_dir = log_dir
        self.frame_per_state=frame_per_state
        self.lstm=lstm
        self.lstm_state=lstm_state
        self.lr = lr
        self.cliprange = cliprange
        ## env model name
        self.env_class=env_class

        ## global model
        self.num_actions = num_actions
        self.ob_dim = ob_dim
        with tf.variable_scope(self.name):
            with tf.device('/device:GPU:0'):
                self.model = Gym_ac_model(num_actions=self.num_actions,
                                          ob_dim=self.ob_dim,
                                          lstm=self.lstm,
                                          scope=self.name)
            ## summary
            self._create_summary()

        ## list of actors
        self.actors = []

        ## create queues
        self.max_predict_batch = self.num_actors
        self.training_q = Queue(maxsize=num_actors)
        self.prediction_q = Queue(maxsize=self.max_predict_batch)
        
        ## share variables
        self.train_step = train_step
        self.predict_counter = predict_counter
        self.last_reward = last_reward
        self.last_length = last_length
        self._flush_train_data()
        ## time to calculate tps/pps
        self.ref_time = time.time()
        self.last_predict_count = 0
        self.last_train_count = 0

    def array_append(self,array1,array2):
        if array1 is None:
            return(array2)
        else:
            return(np.append(array1,array2,axis=0))

    def _flush_train_data(self):
        self.obs_list, self.action_list, self.adv_list, self.vt_list, self.rnn_state = None, None, None, None, None

    def add_actor(self):
        env = self.env_class(self.env_name,image_per_state=self.frame_per_state)
        actor = Actor(len(self.actors),env,self,horizon=self.horizon,lstm=self.lstm)
        self.actors.append(actor)
        self.actors[-1].start()

    def remove_actor(self):
        print("INFO: remove actor")
        while self.actors:
            self.actors[-1].exit_flag.value = True
            self.actors[-1].join()
            self.actors.pop()

    def _create_summary(self):
        ## tensor board related        
        with tf.variable_scope('summary'):
            
            ## create some place holder for scalar tracking
            self.train_q_size = tf.placeholder(dtype=tf.int32,name='train_q_size')
            self.predict_q_size = tf.placeholder(dtype=tf.int32,name='predict_q_size')
            self.reward_ph = tf.placeholder(dtype=tf.int32,name='reward')
            self.length_ph = tf.placeholder(dtype=tf.int32,name='length')
            self.pps = tf.placeholder(dtype=tf.int32,name='predict_per_second')
            self.tps = tf.placeholder(dtype=tf.int32,name='train_per_second')
            
            self.summary_writer = tf.summary.FileWriter("{0}/{1}".format(self.log_dir,self.name),tf.get_default_graph())
            all_summaries = []
            all_summaries.append(tf.summary.scalar('predict_per_second',self.pps))
            all_summaries.append(tf.summary.scalar('train_per_second',self.tps))
            all_summaries.append(tf.summary.scalar('reward',self.reward_ph))
            all_summaries.append(tf.summary.scalar('length',self.length_ph))
            all_summaries.append(tf.summary.scalar('value_loss',self.model.value_loss))
            all_summaries.append(tf.summary.scalar('policy_loss',self.model.policy_loss))
            all_summaries.append(tf.summary.scalar('policy_entropy',self.model.entropy_mean))
            all_summaries.append(tf.summary.scalar('gradient_norm',self.model.orig_norm))
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name):
                all_summaries.append(tf.summary.histogram(var.name,var))
            all_summaries.append(tf.summary.histogram('policy',self.model.policy))
            all_summaries.append(tf.summary.histogram('action',self.model.actions_ph))
            all_summaries.append(tf.summary.histogram('advs',self.model.advs_ph))
            self.merged_summary = tf.summary.merge(all_summaries)


    def _cal_lr_clip(self):
        factor = max(1-self.predict_counter.value/(self.max_train_step),1e-3)
        return(self.lr * factor, self.cliprange * factor)
        

    def train_model(self,feed_dict,sess):
        ## update local variables
        sess.run(self.model.update_local_variables)
        sess.run(self.model.update_pre_variabls)
        
        #self.train_step.value += 1
        lr, cliprange = self._cal_lr_clip()
        feed_dict[self.model.lr] = lr
        feed_dict[self.model.cliprange] = cliprange
        
        if self.train_step.value % 100 == 0 and self.train_step.value > 200:
            feed_dict[self.train_q_size] = self.training_q.qsize()
            feed_dict[self.predict_q_size] = self.prediction_q.qsize()
            feed_dict[self.reward_ph] = self.last_reward.value
            feed_dict[self.length_ph] = self.last_length.value            
            pps,tps = self.pps_tps()
            feed_dict[self.pps] = pps 
            feed_dict[self.tps] = tps
            _,summary = sess.run((self.model.apply_gradient,self.merged_summary) ,feed_dict=feed_dict)
            self.summary_writer.add_summary(summary,self.train_step.value)
        else:
            sess.run(self.model.apply_gradient,feed_dict=feed_dict)

    def pps_tps(self):
        current_time = time.time()
        pps = np.ceil((self.predict_counter.value - self.last_predict_count) / (current_time - self.ref_time))
        tps = np.ceil((self.train_step.value - self.last_train_count) / (current_time - self.ref_time))
        self.ref_time = current_time
        self.last_predict_count = self.predict_counter.value
        self.last_train_count = self.train_step.value
        return(pps,tps)


    def resume_actors(self):
        for actor in self.actors:
            actor.pause.set()


    def predict(self,feed_dict,sess):
        if self.lstm:
            return sess.run((self.model.value,self.model.policy,self.model.rnn_state),feed_dict=feed_dict)
        else:
            return sess.run((self.model.value,self.model.policy),feed_dict=feed_dict)
 
    def run(self,coord=None,sess=None):
             ## firstly add actors
            for _ in range(self.num_actors):
                print('INFO: add actor')
                self.add_actor()
            ## create some predict data place holder
            ## record if all the actors have complete predictions
            actor_dones = np.zeros(self.num_actors,dtype=bool)
            ## create ids array
            ids = np.zeros(self.max_predict_batch,dtype=np.uint16)
            ## obs list array
            obs_predict = np.zeros([self.max_predict_batch]+list(self.ob_dim),dtype=np.float32)
            ## rnn state array
            rnn_state_predict = np.zeros((self.max_predict_batch,self.lstm_state*2),dtype=np.float32)
            while not coord.should_stop():
                ## update local variables
                sess.run(self.model.update_local_variables)
                
                while not actor_dones.all():
                    i = 0
                    while i < self.max_predict_batch:
                        predict_set = self.prediction_q.get(timeout=30)
                        if predict_set['obs'] is None:
                            
                            actor_dones[predict_set['id']]=1
                            break
                        else:
                            ids[i],obs_predict[i]=predict_set['id'],predict_set['obs']
                            if self.lstm:
                                rnn_state_predict[i]=predict_set['state']
                            i += 1
                    if i > 0:
                            ## run model
                        if self.lstm:
                            feed_dict={self.model.cur_obs_ph:obs_predict[:i],self.model.rnn_init:rnn_state_predict[:i]}
                            value_predict,policy_predict,next_state_predict=self.predict(feed_dict,sess)
        
                        else:
                            feed_dict={self.model.cur_obs_ph:obs_predict[:i]}
                            value_predict,policy_predict=self.predict(feed_dict,sess)
                        ## put result back to actor queue
                        for id in range(i):
                            return_set = {'value':value_predict[id],'policy':policy_predict[id]}
                            if self.lstm:
                                return_set['next_state']=next_state_predict[id]
                            self.actors[ids[id]].return_q.put(return_set)
    
                ## reset actor_dones flag
                actor_dones[:] = 0
                ## flush train data
                self._flush_train_data()
                for i in range(self.num_actors):
                    dataset = self.training_q.get(timeout=30)
                    self.obs_list = self.array_append(self.obs_list,dataset['obs'])
                    self.action_list = self.array_append(self.action_list,dataset['action'])
                    self.adv_list = self.array_append(self.adv_list,dataset['adv'])
                    self.vt_list = self.array_append(self.vt_list,dataset['target_value'])                
                    if self.lstm:
                        self.rnn_state=self.array_append(self.rnn_state,dataset['state'])
                ## batch up the data    
                data_size = self.horizon*self.num_actors
                batch_size = data_size // self.mbatch
                for _ in range(self.num_epoch):
                    ## shuffle batch
                    idx = np.arange(data_size)
                    #if not self.lstm:
                    np.random.shuffle(idx)
                    for start in range(0, data_size, batch_size):
                        end = start + batch_size
                        batch_idx = idx[start:end]
                        feed_dict = {
                                self.model.cur_obs_ph:self.obs_list[batch_idx],
                                self.model.actions_ph:self.action_list[batch_idx],
                                self.model.advs_ph:self.adv_list[batch_idx],
                                self.model.target_values_ph:self.vt_list[batch_idx]
                                }
                        if self.lstm:
                            feed_dict[self.model.rnn_init]=self.rnn_state[batch_idx]
                        self.train_model(feed_dict,sess)
                self.train_step.value += 1
                self.resume_actors()

                
        
