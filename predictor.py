import sys
import os
import itertools
import numpy as np
import tensorflow as tf

from gym_atari_ac_model import Gym_ac_model
from utils import discount
from utils import timetest
from multiprocessing import Process, Queue, Value, Event
from datetime import datetime
import time
from threading import Thread

## try not to define it as a subclass of thread yet            
class Predictor(Thread):
    def __init__(self,server,id,num_actions,ob_dim,sess,lstm=True,lstm_state=256):
        #print('INFO: predictor created')
        super(Predictor, self).__init__()
        ## set the predictor to be daemon thread
        self.setDaemon(True)
        self.num_actions = num_actions
        self.ob_dim = ob_dim
        self.exit_flag = False        
        ## define a server
        self.server = server     
        ## tf session
        self.sess = sess                
        ## batch size
        self.batch_size = 128
        
        self.lstm=lstm
        self.lstm_state=lstm_state
        
        
    def run(self):
        
        ## create ids array
        ids = np.zeros(self.batch_size,dtype=np.uint16)
        ## obs list array
        obs_list = np.zeros([self.batch_size]+list(self.ob_dim),dtype=np.float32)
        ## rnn state array
        rnn_state_list = np.zeros((self.batch_size,self.lstm_state*2),dtype=np.float32)

          
        while not self.exit_flag:
            
            
            i = 0
            while i < self.batch_size and (not (i>0 and self.server.prediction_q.empty())):

                predict_set = self.server.prediction_q.get(timeout=30)
                 ids[i],obs_list[i]=predict_set['id'],predict_set['obs']
                if self.lstm:
                    rnn_state_list[i]=predict_set['state']
                i += 1
            ## run model
            if self.lstm:
                feed_dict={self.server.model.cur_obs_ph:obs_list[:i],self.server.model.rnn_init:rnn_state_list[:i]}
                value_list,policy_list,next_state_list=self.server.predict(feed_dict,self.sess)
            else:
                feed_dict={self.server.model.cur_obs_ph:obs_list[:i]}
                value_list,policy_list=self.server.predict(feed_dict,self.sess)
            self.server.predict_counter.value += i
            ## put result back to actor queue
            for id in range(i):
                if ids[id] < len(self.server.actors):
                    return_set = {'value':value_list[id],'policy':policy_list[id]}
                    if self.lstm:
                        return_set['next_state']=next_state_list[id]
                    self.server.actors[ids[id]].return_q.put(return_set)
