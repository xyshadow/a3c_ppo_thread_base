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

## with the thread base approach, the worker get splitted to 3 types, actor, predictor, and trainer

## logger is a process to log results

## actor: actor contains one env, each actor feed observation into a common predict queue, and receive action prodiction from own (depth of 1) result queue.
class Actor(Process):
    def __init__(self,id,env,server,gamma=0.99,n_steps=5,use_pause_flag=True):
        super(Actor, self).__init__()
        self.env = env
        self.server = server
        self.predict_q = self.server.prediction_q
        self.training_q = self.server.training_q
        self.return_q = Queue(maxsize=1)
        self.gamma = gamma
        self.n_steps = n_steps
        
        ## define obs and done
        self.obs = None
        self.done = False
        self.rnn_state = None
        
        ## record reward
        self.total_reward = 0
        self.total_length = 0
        
        ## exit flag, use value as it is shared variable
        self.exit_flag = Value('i',0)
        self.pause = Event()
        
        ## use pause flag or not
        self.use_pause_flag = use_pause_flag
        
        ## id
        self.id = id
        
        ## zero rnn statue
        self.zero_rnn_init = (np.zeros((1,256)),np.zeros((1,256)))
        
        ## num of actions
        self.num_actions = self.env.ac_dim
        
    def _run_one_step(self):
        ## remove dim expand as predictor is now batch
        self.predict_q.put((self.id,self.obs,(self.rnn_state[0][0],self.rnn_state[1][0])))
        ## policy shape should be [num_actions], value should be a scalar, rnn_state should be ([256],[256])
        
        policy, value, rnn_state = self.return_q.get(timeout=10)
        
        action = np.random.choice(np.arange(policy.shape[0]), p=policy)
        next_obs,reward,done,_ = self.env.step(action,self.obs)
        self.total_reward += reward
        self.total_length += 1
        return (action,reward,next_obs,done,value,(np.expand_dims(rnn_state[0],axis=0),np.expand_dims(rnn_state[1],axis=0)))
        
    def run_n_step(self):
        obs_list,action_list,reward_list,next_obs_list,value_list,rnn_states = [],[],[],[],[],[]
        try:
            if self.obs == None:
                self.obs = self.env.reset()
                self.rnn_state = self.zero_rnn_init
        except:
            ValueError

        for _ in range(self.n_steps):
            obs_list.append(self.obs)
            rnn_states.append(self.rnn_state)
            action,reward,next_obs,self.done,value,next_rnn_state=self._run_one_step()
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            value_list.append(value)
            
            if self.done:
                ## reset env
                self.obs = self.env.reset()
                self.rnn_state = self.zero_rnn_init

                ## write reward and length to server variables
                self.server.last_reward.value = int(self.total_reward)
                self.server.last_length.value = int(self.total_length)
                self.total_length = 0
                self.total_reward = 0
                break
            else:
                self.obs = next_obs
                self.rnn_state = next_rnn_state
        
        ## caculate ref values and adv
        if self.done:
            value_p1 = 0
        else:
            self.predict_q.put((self.id,self.obs,(self.rnn_state[0][0],self.rnn_state[1][0])))
            _, value_p1, _ = self.return_q.get()
            
        rewards_plus = np.asarray(reward_list + [value_p1])
        reward_list = discount(rewards_plus,self.gamma)[:-1]
        value_plus = np.asarray(value_list + [value_p1])
        advs = rewards_plus[:-1] + self.gamma * value_plus[1:] - value_plus[:-1]
        lamb = 1
        advs = discount(advs,self.gamma * lamb)
        self.training_q.put((np.array(obs_list),np.array(action_list),advs,reward_list,rnn_states[0]))

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        ## 
        while self.exit_flag.value == 0:
            self.run_n_step()
            if self.use_pause_flag:
                self.pause.clear()
                self.pause.wait(timeout=60)
                

## try not to define it as a subclass of thread yet            
class Predictor(Thread):
    def __init__(self,server,id,num_actions,sess):
        super(Predictor, self).__init__()
        ## set the predictor to be daemon thread
        self.setDaemon(True)
        
        self.exit_flag = False        
        ## define a server
        self.server = server     
        self.sess = sess                
       ## record last train count
        self.last_train_count = 0
        ## batch size
        self.batch_size = 128
        
    def run(self):
        ## create ids array
        ids = np.zeros(self.batch_size,dtype=np.uint16)
        ## obs list array
        obs_list = np.zeros((self.batch_size,84,84,1),dtype=np.float32)
        ## rnn state array
        rnn_state_list = (np.zeros((self.batch_size,256),dtype=np.float32),np.zeros((self.batch_size,256),dtype=np.float32))
        ## mask_batch
        ## seq_len
        seq_len_batch = np.ones((self.batch_size),dtype=np.int32)
            
        while not self.exit_flag:
            
            ## get task from queue
            ## obs shape: 84X84Xframe rnn_state shape (256,256)
            q_return = self.server.prediction_q.get(timeout=10)
            ## data shape cheking
            assert type(q_return[0]) is int, 'error: return id is not type int'
            assert q_return[1].shape == (84,84,1), 'error: return obs is with wrong shape:{0}'.format(q_return[1].shape)
            assert q_return[2][0].shape == (256,) and q_return[2][1].shape == (256,), 'error: return rnn state is with wrong shape:{0},{1}'.format(q_return[2][0].shape, q_return[2][1].shape)
            ids[0], obs_list[0], (rnn_state_list[0][0],rnn_state_list[1][0]) = q_return
            size = 1
            while size < self.batch_size and not self.server.prediction_q.empty():
                ids[size], obs_list[size], (rnn_state_list[0][size],rnn_state_list[1][size]) = self.server.prediction_q.get(timeout=10)
                size += 1
            
            
            ## run model
            rnn_state_batch = (rnn_state_list[0][:size],rnn_state_list[1][:size])
            feed_dict={self.server.model.cur_obs_ph:obs_list[:size],
                       self.server.model.rnn_init:rnn_state_batch,
                       self.server.model.seq_len_ph:seq_len_batch[:size],
                       self.server.model.batch_size:size,
                       self.server.model.seq_len:1}
            ## policy shape should be batchXnum_actions, value should be batchX1, rnn_state should be (batchX256,batchX256)
            
            policy,value,next_rnn_state=self.server.predict(feed_dict,self.sess)
            self.server.predict_counter.value += size
            ## put result back to actor queue
            for i in range(size):
                if ids[i] < len(self.server.actors):
                    self.server.actors[ids[i]].return_q.put((policy[i],value[i],(next_rnn_state[0][i],next_rnn_state[1][i])))
                

class Trainer(Thread):
    def __init__(self, server, id, sess, async_training=False, batch_size=16):
        super(Trainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False
        self.sess = sess
        self.batch_size = batch_size
        ## training type can be either sync or async
        self.async_training = async_training
        
        ## hard code mini batch size and epoch
        self.mbatch = 4
        self.epoch = 4

    def run(self):
        ## create batch place holder
        obs_batch = np.zeros((self.batch_size*self.server.n_steps,84,84,1),dtype=np.float32)
        actions_batch = np.zeros((self.batch_size*self.server.n_steps),dtype=np.int32)
        advs_batch = np.zeros((self.batch_size*self.server.n_steps),dtype=np.float32)
        value_batch = np.zeros((self.batch_size*self.server.n_steps),dtype=np.float32)
        rnn_state_batch = (np.zeros((self.batch_size,256),dtype=np.float32),np.zeros((self.batch_size,256),dtype=np.float32))
        mask_batch = np.zeros((self.batch_size*self.server.n_steps),dtype=np.int32)
        seq_len_batch = np.zeros((self.batch_size),dtype=np.int32)
        
        while not self.exit_flag:
            size = 0
            ## when size is 0, always wait
            while size < self.batch_size and (not self.async_training or not (size>0 and self.server.training_q.empty())):
                obs_list,action_list,advs,reward_list,rnn_states = self.server.training_q.get(timeout=20)
                ## obtain episode length
                ep_l = action_list.shape[0]
                obs_batch[size*self.server.n_steps:size*self.server.n_steps+ep_l]=obs_list
                actions_batch[size*self.server.n_steps:size*self.server.n_steps+ep_l]=action_list
                advs_batch[size*self.server.n_steps:size*self.server.n_steps+ep_l]=advs
                value_batch[size*self.server.n_steps:size*self.server.n_steps+ep_l]=reward_list
                rnn_state_batch[0][size] = rnn_states[0]
                rnn_state_batch[1][size] = rnn_states[1]
                mask_batch[size*self.server.n_steps:size*self.server.n_steps+ep_l]=np.ones((ep_l),dtype=np.int32)
                seq_len_batch[size] = ep_l
                size += 1
                
            for _ in range(self.epoch):
                for idx in range(size//self.mbatch):
                    temp_rnn_state_batch = (rnn_state_batch[0][idx:idx+self.mbatch],rnn_state_batch[1][idx:idx+self.mbatch])
                    r_from = idx*self.mbatch*self.server.n_steps
                    r_to = (idx+1)*self.mbatch*self.server.n_steps
                    feed_dict = {
                            self.server.model.cur_obs_ph:obs_batch[r_from:r_to],
                            self.server.model.actions_ph:actions_batch[r_from:r_to],
                            self.server.model.advs_ph:advs_batch[r_from:r_to],
                            self.server.model.values_ph:value_batch[r_from:r_to],
                            self.server.model.rnn_init:temp_rnn_state_batch,
                            self.server.model.mask_ph:mask_batch[r_from:r_to],
                            self.server.model.seq_len_ph:seq_len_batch[idx:idx+self.mbatch],
                            self.server.model.batch_size:self.mbatch,
                            self.server.model.seq_len:self.server.n_steps}
                    self.server.train_counter.value += self.mbatch
                    self.server.train_model(feed_dict,self.sess)
                    self.server.resume_actors()

