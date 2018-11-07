import numpy as np
import tensorflow as tf
import gym
import gym.spaces
from gym.wrappers import Monitor

from utils import *


 
class Gym_atari_env(object):
    def __init__(self,env_name,max_path_length=None,video=False,video_dir=None,image_per_state=4):
        ## arguments
        ## env_name: name of the gym env
        ## max_path_length: max path length before a hard reset, if None, just use env default
        print("INFO: creating new env of %s" % env_name)
        ## create a new env
        self.env = gym.make(env_name)
        # remove the timelimitwrapper
        #self.env = env.env
        self.image_per_state=image_per_state
        ## define max path, either from user input or env default
        self.max_path_length = max_path_length or self.env.spec.max_episode_steps
        
        ## define parameters directly
        self.discrete =True
        ##all image will be processed to the dimension below
        self.ob_dim = [84,84,self.image_per_state]
        self.ac_dim = 4 if env_name == "Pong-v0" or env_name == "Breakout-v0" else self.env.action_space.n
        
        ## if video, use monitor
        if video:
            self.env = Monitor(self.env, directory=video_dir, video_callable=lambda x: True, resume=True)
        ## init process image
        #self.process_image
    
    #@define_scope
    def process_image(self,raw_image):
    
        ## grep scale
        out = np.mean(raw_image, axis=2).astype(np.uint8)
        ## crop, down sample and normalise
        out = out[34:194,:][::2, ::2] / 255
        ## dim expand
        out = np.hstack([np.zeros((80,2)),out,np.zeros((80,2))])
        out = np.vstack([np.zeros((2,84)),out,np.zeros((2,84))])

        return out

    def step(self,action,obs,sess=None,training=True):
        ## arguments
        ## action: action to take        
        ## return ob, rew, done, info
 
        raw_image,rew,done,info = self.env.step(action)
        next_obs = self.process_image(raw_image)
        next_obs = np.append(obs[:,:,1:], np.expand_dims(next_obs, 2), axis=2)
        
        return (next_obs,rew,done,info)
    
    def reset(self,sess=None):
        raw_image = self.env.reset()
        obs = self.process_image(raw_image)
        return np.stack([obs] * self.image_per_state, axis=2)
    
    def statistic(self):
        return self.discrete, self.max_path_length, self.ob_dim, self.ac_dim

