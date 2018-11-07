import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile

from gym.wrappers import Monitor
import gym

from gym_atari_ac_model import Gym_ac_model

class Run_monitor(object):
    
    def __init__(self,env,global_step,global_model,saver=None,cache_dir=None):
        
        self.video_dir = os.path.join(cache_dir, "videos")
        self.video_dir = os.path.abspath(self.video_dir)
        ## create a env with video monitor
        self.env=env
        num_actions=self.env.ac_dim
        self.cache_dir=cache_dir

        ## create a model
        with tf.variable_scope('monitor'):
            self.model = Gym_ac_model(num_actions=num_actions,
                                      global_train_op=global_model,
                                      scope='monitor')
            self._create_summary()
        #self.summary_writer = summary_writer
        self.global_step = global_step
        
        ## create a check point saver
        #self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)
        self.saver = saver
        if self.saver is not None:
            
            ## create a saver dir
            self.checkpoint_path = os.path.join(cache_dir, "checkpoints")
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
        ## video path
        try:
            os.makedirs(self.video_dir)
        except FileExistsError:
            pass

    ## create summary
    def _create_summary(self):
        ## tensor board related        
        with tf.variable_scope('summary'):
            self.summary_writer = tf.summary.FileWriter("{0}/{1}".format(self.cache_dir,'monitor'),tf.get_default_graph())
            all_summaries = []
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'monitor'):
                all_summaries.append(tf.summary.histogram(var.name,var))
            
            
            self.merged = tf.summary.merge(all_summaries)

    

    @property
    def zero_rnn_init(self):
        return (np.zeros(self.model.rnn_init[0].get_shape().as_list()),np.zeros(self.model.rnn_init[1].get_shape().as_list()))

        
    def eval_once(self,sess):
         with sess.as_default(), sess.graph.as_default():
             sess.run(self.model.update_local_variables)
             done = False
             total_reward = 0
             episode_length = 0
             obs = self.env.reset(sess)
             rnn_state = self.model.zero_rnn_init

             while not done:
                 policy,value,rnn_state=sess.run((self.model.policy,self.model.value,self.model.rnn_state),
                                       feed_dict={self.model.cur_obs_ph:np.expand_dims(obs,axis=0),self.model.rnn_init:rnn_state})
                 action = np.random.choice(np.arange(policy.shape[1]), p=policy[0])
                 ## make one action
                 next_obs, reward, done, _ = self.env.step(action,obs,sess,training=False)
                 total_reward += reward
                 episode_length += 1
                 obs = next_obs
                
             # Add summaries
             episode_summary = tf.Summary()
             episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
             episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
             metrics_summary, step = sess.run((self.merged,tf.train.get_global_step()))
             self.summary_writer.add_summary(episode_summary, step)
             self.summary_writer.add_summary(metrics_summary, step)
             self.summary_writer.flush()
                
             if self.saver is not None:
                 self.saver.save(sess, self.checkpoint_path+'/model')

                
             
    def continuous_eval(self, eval_every, sess, coord):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return
             
