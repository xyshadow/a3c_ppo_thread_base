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

## actor: actor contains one env, each actor feed observation into a common predict queue, and receive action prodiction from own (depth of 1) result queue.
class Actor(Process):
    def __init__(self,id,env,server,gamma=0.99,lambd=0.95,horizon=128,lstm=True,lstm_state=256,use_pause_flag=True):
        super(Actor, self).__init__()
        ## id
        self.id = id
        self.env = env
        self.server = server
        self.gamma = gamma
        self.lambd = lambd
        self.horizon = horizon
        self.lstm=lstm
        self.lstm_state=256
        ## use pause flag or not
        self.use_pause_flag = use_pause_flag        

                
        ## queues
        self.predict_q = self.server.prediction_q
        self.training_q = self.server.training_q
        self.return_q = Queue(maxsize=1)

        ## zero rnn statue, just keep it as 2x of self.lstm_state, and format gets transfered within the model
        self.zero_rnn_init = np.zeros(self.lstm_state*2,'float32')

       ## num of actions
        self.num_actions = self.env.ac_dim

        ## define status as global variables
        self.obs = self.env.reset()
        if self.lstm:
            self.rnn_state = self.zero_rnn_init
       
        ## record reward
        self.total_reward = 0
        self.total_length = 0
        
        ## exit flag, use value as it is shared variable
        self.exit_flag = Value('i',0)
        self.pause = Event()
        
        ## use pause flag or not
        self.use_pause_flag = use_pause_flag
        
    def _cal_vt_adv(self,done_list,value_list,reward_list,next_v):
        """
        based on openai baseline ppo1
        """
        done_list = np.append(done_list,0)
        value_list = np.append(value_list,next_v)
        t = len(reward_list)
        adv_list = np.zeros(t,'float32')
        lastgaelam = 0
        for i in reversed(range(t)):
            nonterminal = 1-done_list[i+1]
            ## if done is true, value+1 is 0
            delta = reward_list[i] + self.gamma * value_list[i+1] * nonterminal - value_list[i]
            adv_list[i]=lastgaelam=delta + self.gamma * self.lambd * nonterminal * lastgaelam
            vt_list=adv_list+value_list[:-1]
        adv_list = (adv_list - adv_list.mean()) / (adv_list.std() + 1e-8)
        return adv_list,vt_list
            
         
    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        
        ##initialize record variables
        obs_list = np.array([self.obs for _ in range(self.horizon)])
        reward_list = np.zeros(self.horizon, 'float32')
        value_list = np.zeros(self.horizon, 'float32')
        action_list = np.zeros(self.horizon, 'int32')
        done_list = np.zeros(self.horizon, 'int32')
        if self.lstm:        
            rnn_states_list = np.array([self.rnn_state for _ in range(self.horizon)])
        t = 0
        done=True
        self.obs=self.env.reset()
        self.rnn_state = self.zero_rnn_init
        while self.exit_flag.value == 0:
            obs_set = {'id':self.id,'obs':self.obs}
            if self.lstm:
                obs_set['state'] = self.rnn_state
            self.predict_q.put(obs_set)
            if self.id == 0:
                self.server.predict_counter.value +=1
            result_set = self.return_q.get(timeout=10)
            value, policy = result_set['value'], result_set['policy']
            if self.lstm:
                next_rnn_state = result_set['next_state']
            
            ## check if horizon step is met after taking one step, so the value+1 is always available by default
            if t >= self.horizon:
                ## reset t
                t = 0
                ## work out the adv
                adv_list,vt_list = self._cal_vt_adv(done_list,value_list,reward_list,value*(1-done))
                ## send training data
                train_data = {'obs':obs_list,'action':action_list,'adv':adv_list,'target_value':vt_list}
                if self.lstm:
                    train_data['state']=rnn_states_list
                ## tell server prediction is done
                self.predict_q.put({'id':self.id,'obs':None})
                self.training_q.put(train_data)
                self.pause.clear()

                self.pause.wait()
                
            
            obs_list[t]=self.obs
            value_list[t]=value
            action_list[t]=np.random.choice(np.arange(policy.shape[0]), p=policy,)
            done_list[t]=done
            if self.lstm:
                rnn_states_list[t]=self.rnn_state
            ## take one action
            next_obs,reward,done,_ = self.env.step(action_list[t],self.obs)
            ## record reward
            reward_list[t]=reward
            
            
            self.total_reward += reward
            self.total_length += 1
                        

            if done:
                ## reset env
                self.obs = self.env.reset()
                if self.lstm:
                    ## reset state
                    self.rnn_state = self.zero_rnn_init
                ## record reward and length
                self.server.last_reward.value = int(self.total_reward)
                self.server.last_length.value = int(self.total_length)
                self.total_length = 0
                self.total_reward = 0
            else:                
                ## update current obs
                self.obs=next_obs
                if self.lstm:
                    ## update current state
                    self.rnn_state=next_rnn_state
            
            t += 1
            
