import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *

def lander_model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def lander_optimizer(lr=1e-3):
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(lr),
        kwargs={}
    )

def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    return stopping_criterion

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def lander_kwargs(optimizer):
    return {
        'optimizer_spec': optimizer,
        'q_func': lander_model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def lander_learn(env,
                 session,
                 num_timesteps,
                 seed,
                 lr=1e-3,
                 double_q=True):

    optimizer = lander_optimizer(lr=lr)
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)


    returns = dqn.learn(
        env=env,
        session=session,
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        double_q=double_q,
        **lander_kwargs(optimizer)
    )
    env.close()

    return returns

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count={'GPU': 0})
    # GPUs don't significantly speed up deep Q-learning for lunar lander,
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    return session

def get_env(seed):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env

def main(lr=1e-3, double_q=True):
    # Run training
    seed = 4565 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(seed)
    session = get_session()
    set_global_seeds(seed)
    returns = lander_learn(env, session, num_timesteps=500000, seed=seed, lr=lr, double_q=double_q)

    return returns

def sweep_lr():
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]

    fig = plt.figure()
    div = 10000
    for lr in lrs:
        t, mean_reward, _ = main(lr=lr)
        t = [t_/div for t_ in t]
        plt.plot(t, mean_reward, label=str(lr))

    fig.legend(loc="best") 
    plt.xlabel("x%d iteration" % div)
    plt.show()

def double_vs_vanilla_q():
    double_q_list = [False, True]
    labels = ["Vanilla DQN", "Double DQN"]

    fig = plt.figure()
    div = 10000
    for double_q, label in zip(double_q_list, labels):
        t, mean_reward, _ = main(double_q=double_q)
        t = [t_/div for t_ in t]
        plt.plot(t, mean_reward, label=label)

    fig.legend(loc="best")
    plt.xlabel("x%d iteration" % div)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_lr", action="store_true")
    parser.add_argument("--double_vs_vanilla_q", action="store_true")
    args = parser.parse_args()

    if args.sweep_lr:
        sweep_lr()
    elif args.double_vs_vanilla_q:
        double_vs_vanilla_q()
    else:
        main()
