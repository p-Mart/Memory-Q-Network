# This file is the main entry point into the tensorflow version
# of this research project.

import sys, os, errno
import itertools as it
import random
from tqdm import trange

import numpy as np
import tensorflow as tf
import gym
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import *
from vizdoom import *

from utilities import *

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), (size, 5))

class DQN:
    def __init__(self, learning_rate, nb_actions, observation_shape, h_size=128):
        self.input = tf.placeholder(shape=[None]+list(observation_shape), dtype=tf.float32)

        # Architecture
        self.conv_1 = tf.layers.conv2d(
            inputs=self.input,
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.conv_2 = tf.layers.conv2d(
            inputs=self.conv_1,
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.conv_3 = tf.layers.conv2d(
            inputs=self.conv_1,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.flatten = tf.layers.flatten(self.conv_3)

        self.h_1 = tf.layers.dense(
            inputs=self.flatten,
            units=h_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.output =tf.layers.dense(
            inputs=self.h_1,
            units=nb_actions
        )

        self.predict = tf.argmax(self.output, axis=1)

        # Everything needed to calculate loss function
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32)

        self.q = tf.reduce_sum(tf.multiply(self.output, self.actions_onehot), axis=1)

        self.loss = tf.losses.mean_squared_error(self.target_q, self.q)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_model = self.optimizer.minimize(self.loss)

def updateTargetGraph(tf_vars, target_update):
    total_vars = len(tf_vars) // 2
    op_holder = []
    for idx, var in enumerate(tf_vars[0:total_vars]):
        value = var.value()*target_update + (1 - target_update)*tf_vars[idx+total_vars].value()
        op_holder.append(tf_vars[idx+total_vars].assign(value))
    return op_holder

def updateTargetNetwork(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def preprocess(data, resolution):
    """
    Data processor

    :param data: game frames
    :param resolution: resolution to resize game frames to
    :return: resized image
    """

    processed_data = skimage.transform.resize(data, output_shape=resolution)
    processed_data = processed_data.astype(np.float32) / 255.

    return processed_data

def main(model_name, options):
    # Hyperparameters
    batch_size = 64
    update_freq = 4
    save_freq = 100
    target_model_update = 0.001
    start_eps = 1.0
    end_eps = 0.1
    learning_rate = 0.00025
    discount = 0.99
    replay_size = 100000
    nb_steps_warmup = 10000
    nb_episodes = 1000
    nb_episodes_test = 100
    h_size = 512
    buffer_size = 10000
    env_name = 'Pong-v0'

    hyperparameters = {
        'batch_size' : batch_size,
        'update_freq' : update_freq,
        'save_freq' : save_freq,
        'target_model_update' : target_model_update,
        'start_eps' : start_eps,
        'end_eps' : end_eps,
        'learning_rate' : learning_rate,
        'discount' : discount,
        'replay_size': replay_size,
        'nb_steps_warmup' : nb_steps_warmup,
        'nb_episodes' : nb_episodes,
        'nb_episodes_test' : nb_episodes_test,
        'h_size' : h_size,
        'buffer_size' : buffer_size,
        'env_name' : env_name
    }

    # Initialize environment.
    env = gym.make(env_name)
    #wrapper = ToDiscrete('minimal')
    #env = wrapper(env)

    # Env parameters
    nb_actions = env.action_space.n
    #observation_shape = env.observation_space.shape # Might break?
    processed_resolution = (210 / 3,160 / 3, 1)
    observation_shape = processed_resolution

    # Initialize networks, tensorflow vars
    tf.reset_default_graph()

    model = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape
    )

    target_model = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape
    )

    # Tensorflow Stuff
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter(logdir="./logs/{}".format(model_name),graph=sess.graph)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    target_ops = updateTargetGraph(trainables, target_model_update)
    experience = ReplayBuffer(buffer_size=buffer_size)

    sess.run(init)

    # Other parameters
    eps = start_eps
    if "load" in options:
        eps = end_eps

    eps_stepdown = (start_eps - end_eps) / nb_steps_warmup

    steps_per_episode = []
    reward_list = []
    total_steps = 0

    # tensorboard = TensorBoard(log_dir="./logs/{}".format(model_name))
    # Enable debugging utility
    if "debug" in options:
        pass
        #callbacks.append(LayerDebug())

    log_dir = "./data/{}/".format(model_name)
    # Weights will be loaded if weight file exists.
    if os.path.exists(log_dir) and "load" in options:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

    # Attempt to create log directory for this model.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Training loop, if enabled
    if "train" in options:
        episode = 0
        try:
            for episode in trange(nb_episodes):
                observation = env.reset()
                observation = preprocess(observation, processed_resolution)
                done = False
                r_all = 0
                steps = 0

                while not done:
                    steps += 1
                    total_steps += 1

                    # Take random action with probability eps
                    if random.random() < eps or total_steps < nb_steps_warmup:
                        action = random.randint(0, nb_actions - 1)
                    else:
                        action = sess.run(model.predict, feed_dict={model.input:[observation]})[0]

                    # Step through environment
                    observation_next, reward, done, _ = env.step(action)

                    # Record in experience buffer
                    observation_next = preprocess(observation_next, processed_resolution)
                    new_experience = np.array([observation, action, reward, observation_next, done])
                    new_experience = new_experience.reshape((1, 5))
                    experience.add(new_experience)

                    if total_steps > nb_steps_warmup:
                        if eps > end_eps:
                            eps -= eps_stepdown
                        if total_steps % update_freq == 0:
                            train_batch = experience.sample(batch_size)

                            s0_batch = np.stack(train_batch[:, 0], axis=0)
                            s1_batch = np.stack(train_batch[:, 3], axis=0)

                            model_q = sess.run(model.predict, feed_dict={model.input:s1_batch})
                            target_model_q = sess.run(target_model.output, feed_dict={target_model.input:s1_batch})

                            end_multiplier = 1 - train_batch[:, 4]
                            double_q = target_model_q[range(batch_size), model_q]
                            target_q = train_batch[:, 2] + (discount * double_q * end_multiplier)

                            # Update model
                            _ = sess.run(model.update_model, feed_dict={
                                model.input : np.stack(s0_batch, axis=0),
                                model.target_q : target_q,
                                model.actions : train_batch[:, 1]
                            })

                            # Update target network
                            updateTargetNetwork(target_ops, sess)



                    r_all += reward
                    observation = observation_next

                    if done:
                        break

                # Log reward in tensorboard
                # TODO : Make this into a function pls
                episode_reward_summary = tf.Summary()
                reward_value = episode_reward_summary.value.add()
                reward_value.simple_value = r_all
                reward_value.tag = 'reward'
                summary_writer.add_summary(episode_reward_summary, episode)
                summary_writer.flush()

                steps_per_episode.append(steps)
                reward_list.append(r_all)

                # Save model occasionally
                if episode % save_freq == 0:
                    saver.save(sess, log_dir+'/model-'+str(episode+1)+'.ckpt')
                if len(reward_list) % 10 == 0:
                    print(total_steps, np.mean(reward_list[-10:]), eps)
        except KeyboardInterrupt:
            print("Training interrupted manually.")

        # Save model
        saver.save(sess, log_dir + '/model-' + str(episode+1) + '.ckpt')

        # Log all hyperparameter settings
        logHyperparameters(model_name, **hyperparameters)

    # Test loop, if enabled
    if "test" in options:
        episode = 0
        try:
            for episode in trange(nb_episodes_test):
                observation = env.reset()
                observation = preprocess(observation, processed_resolution)
                done = False
                r_all = 0
                steps = 0

                while not done:
                    steps += 1
                    total_steps += 1

                    # Only the network is used to take an action
                    action = sess.run(model.predict, feed_dict={model.input:[observation]})[0]
                    print(action)
                    # Step through environment
                    env.render()
                    observation, reward, done, _ = env.step(action)
                    observation = preprocess(observation, processed_resolution)

                    # Store reward
                    r_all += reward

                    if done:
                        break

                # Log reward in tensorboard
                # TODO : Make this into a function pls
                episode_reward_summary = tf.Summary()
                reward_value = episode_reward_summary.value.add()
                reward_value.simple_value = r_all
                reward_value.tag = 'reward_test'
                summary_writer.add_summary(episode_reward_summary, episode)
                summary_writer.flush()

                steps_per_episode.append(steps)
                reward_list.append(r_all)
        except KeyboardInterrupt:
            print("Testing interrupted manually.")

    return


if __name__ == "__main__":

    model_name = None
    options = []

    if len(sys.argv) >= 3:
        model_name = sys.argv[1]

        mode = sys.argv[2]

        if mode != "train" and mode != "test":
            print("Usage: python MQN.py [model_name].h5 [train | test]")
            sys.exit()

        options = sys.argv[2:]

    else:
        print("Incorrect number of arguments.")
        print("Usage: python MQN.py [model_name].h5 [train | test]")
        sys.exit()

    main(model_name, options)