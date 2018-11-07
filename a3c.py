#!/usr/bin/env python
from multiprocessing import Queue, Value, Event

import os
import itertools
import numpy as np
import tensorflow as tf
import threading
import time

from gym_atari_ac_model import Gym_ac_model
from gym_atari_env import Gym_atari_env
from actor import Actor
from server import Server
import argparse

def run_episode(sess,model,env,summary_writer,step,lstm=True,lstm_state=256):
    obs = env.reset()
    done = False
    rnn_state = np.zeros((1,lstm_state*2),'float32')
    score = 0
    length = 0
    while not done:
        if lstm:
            policy,rnn_state = sess.run((model.policy,model.rnn_state),feed_dict={model.cur_obs_ph:np.expand_dims(obs,0),model.rnn_init:rnn_state})
        else:
            policy = sess.run((model.policy),feed_dict={model.cur_obs_ph:obs})
    
        obs,reward,done,_ = env.step(np.argmax(policy),obs,training=False)
        score += reward
        length += 1
    env.reset()
    ## write to summary
    summary=tf.Summary()
    summary.value.add(simple_value=score, tag="total_reward")
    summary.value.add(simple_value=length, tag="total_length")
    summary_writer.add_summary(summary, step)
    summary_writer.flush()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',type=str, default='Pong-v0')
    parser.add_argument('--max_train_step',type=int, default=1e7)
    parser.add_argument('--model_dir',type=str, default='./train')
    parser.add_argument('--eval_every',type=int, default=300)
    parser.add_argument('--learning_rate',type=float, default=2.5e-4)
    parser.add_argument('--num_actor',type=int, default=8)
    parser.add_argument('--num_server',type=int, default=2)
    parser.add_argument('--horizon',type=int, default=128)
    parser.add_argument('--mbatch_size',type=int, default=4)
    parser.add_argument('--num_epoch',type=int, default=3)
    parser.add_argument('--frame_per_state',type=int, default=1)
    args = parser.parse_args()
    
    
    
    ## create a global model on cpu
    env = Gym_atari_env(args.env_name,image_per_state=args.frame_per_state,video=True,video_dir=os.path.join(args.model_dir, "videos"))

    num_actions = env.ac_dim
    ob_dim = env.ob_dim
    with tf.variable_scope('global'):
        with tf.device('/device:GPU:0'):
            model = Gym_ac_model(num_actions=num_actions,
                                 ob_dim=ob_dim,
                                 scope='global',
                                 lstm=True,
                                 trainable=False)
    ## define servers
    servers = []
    num_servers = args.num_server
    for id in range(num_servers):
        server = Server(name='server{}'.format(id),
                        env_name=args.env_name,
                        max_train_step=args.max_train_step,
                        num_actors=args.num_actor,
                        horizon=args.horizon,                 
                        mbatch=args.mbatch_size,
                        num_epoch=args.num_epoch,
                        log_dir=args.model_dir,
                        num_actions=num_actions,
                        ob_dim=ob_dim,
                        frame_per_state=args.frame_per_state,
                        lr = args.learning_rate)
        servers.append(server)
    ## define a saver
    saver = tf.train.Saver(var_list=model.global_vars, keep_checkpoint_every_n_hours=2.0, max_to_keep=10)
    ## create a saver dir
    checkpoint_path = os.path.join(args.model_dir, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## define a summary to write eval results
    summary_writer = tf.summary.FileWriter("{0}/eval".format(args.model_dir),tf.get_default_graph())

    ## initilise
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        ## load check point if exist
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print("Loading model checkpoint: {}".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        coord = tf.train.Coordinator()
        server_threads = []
        for server in servers:
            # randomly sleep up to 1 second. helps agents boot smoothly.
            time.sleep(np.random.rand())

            t = threading.Thread(target=server.run,args=(coord,sess),name=server.name)
            t.start()
            print('INFO:{} start'.format(server.name))
        server_threads.append(t)
        print('INFO:training start')
        
        while servers[0].predict_counter.value < args.max_train_step:
            time.sleep(600)
            saver.save(sess,checkpoint_path+'/model')
            ## create video
            run_episode(sess,model,env,summary_writer,servers[0].predict_counter.value)
       ## shut down
        print('INFO: training complete')

        for server in servers:
            server.remove_actor()
        coord.request_stop()        
        
        coord.join(server_threads)
