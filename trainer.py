import sys
import os
import itertools
import numpy as np
import tensorflow as tf
import time

from gym_atari_ac_model import Gym_ac_model
from utils import discount
from utils import timetest
from multiprocessing import Process, Queue, Value, Event
from datetime import datetime
import time
from threading import Thread

class Trainer(Thread):
    def __init__(self,server,id,num_actions,ob_dim,sess,lstm=True,horizon=128,num_actors=8,mbatch=4,epoch=3):
        super(Trainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False
        self.sess = sess
        self.lstm=lstm
        self.num_actions = num_actions
        self.ob_dim = ob_dim
        self.horizon=horizon
        self.num_actors=num_actors
        self.mbatch = mbatch
        self.epoch = epoch
        self._flush_train_data()

    def array_append(self,array1,array2):
        if array1 is None:
            return(array2)
        else:
            return(np.append(array1,array2,axis=0))

    def _flush_train_data(self):
        self.obs_list, self.action_list, self.adv_list, self.vt_list, self.rnn_state = None, None, None, None, None

    def run(self):
        
        while not self.exit_flag:
            
            self._flush_train_data()
            i = 0
            while i < self.num_actors:
                dataset = self.server.training_q.get(timeout=30)
                self.obs_list = self.array_append(self.obs_list,dataset['obs'])
                self.action_list = self.array_append(self.action_list,dataset['action'])
                self.adv_list = self.array_append(self.adv_list,dataset['adv'])
                self.vt_list = self.array_append(self.vt_list,dataset['target_value'])                
                if self.lstm:
                    self.rnn_state=self.array_append(self.rnn_state,dataset['state'])
                i += 1
            data_size = self.horizon*self.num_actors
            batch_size = data_size // self.mbatch
            for _ in range(self.epoch):
                ## shuffle batch
                idx = np.arange(data_size)
                np.random.shuffle(idx)
                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    batch_idx = idx[start:end]
                    feed_dict = {
                            self.server.model.cur_obs_ph:self.obs_list[batch_idx],
                            self.server.model.actions_ph:self.action_list[batch_idx],
                            self.server.model.advs_ph:self.adv_list[batch_idx],
                            self.server.model.target_values_ph:self.vt_list[batch_idx]
                            }
                    if self.lstm:
                        feed_dict[self.server.model.rnn_init]=self.rnn_state[batch_idx]
                    self.server.train_counter.value += 1
                    self.server.train_model(feed_dict,self.sess)
            self.server.resume_actors()


