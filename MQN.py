import sys, os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.callbacks import TrainEpisodeLogger, TestLogger

import gym
#from openai_maze_envs import gym_maze
from MQNModel import MQNmodel
from DQNModel import DQNmodel

#Need to update environment to have indicator tiles
#Need better metrics (percent of episodes)
#Need model saving / loading
#Need model visualization ability

def visualizeLayer(model, layer, sample):
    inputs = [K.learning_phase()] + model.inputs

    _output = K.function(inputs, [layer.output])
    def output(x):
        return _output([0] + [x])

    output = output(sample)
    output = np.squeeze(output)

    n = int(np.ceil(np.sqrt(output.shape[0])))
    fig = plt.figure(figsize=(12,8))
    for i in range(output.shape[0]):
        ax = fig.add_subplot(n,n,i+1)
        im = ax.imshow(output[i], cmap='jet')

    #plt.imshow(output[0], cmap='jet')
    cbar_ax = fig.add_axes([0.15,0.05,0.7,0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    plt.show()

    #print output
    #print "Output shape: ", output.shape


def showmetrics(log):
    ##### Metrics #####
    episodic_reward_means = []
    episodic_losses = []
    episodic_mean_q = []

    for key, value in log.rewards.items():
        episodic_reward_means.append(np.mean(log.rewards[key]))

    for key, value in log.episodic_metrics_variables.items():
        for i in range(0, len(log.episodic_metrics_variables[key]), 2):
            name = log.episodic_metrics_variables[key][i]
            val = log.episodic_metrics_variables[key][i+1]
            if(name == "loss" and val != '--'):
                episodic_losses.append(val)
            if(name == "mean_q" and val != '--'):
                episodic_mean_q.append(val)

    #Windowed Mean
    #episodic_reward_means = (np.cumsum(episodic_reward_means)  
    #                                           / (np.arange(len(episodic_reward_means)) + 1))


    window_size = 50
    averaged_rewards = []
    for i in range(0, len(episodic_reward_means), window_size):

        window = episodic_reward_means[i:i+window_size]
        temp_sum = np.sum(episodic_reward_means[i:i+window_size])

        averaged_rewards.append(temp_sum / window_size)


    plt.figure(1)
    plt.subplot(311)
    plt.title("Loss per Episode")
    plt.plot(episodic_losses, 'b')

    plt.subplot(312)
    plt.title("Mean Q Value per Episode")
    plt.plot(episodic_mean_q, 'r')

    plt.subplot(313)
    plt.title("Reward")
    plt.plot(averaged_rewards, 'g')
    plt.show()

def main(weights_file):
    #Initialize maze environments.
    #env1 = gym.make('IMaze2-v0')
    #env2 = gym.make('IMaze3-v0')
    #env3 = gym.make('IMaze6-v0')
    env4 = gym.make('IMaze8-v0')

    envs = [env4]

    #Setting hyperparameters.
    nb_actions = env4.action_space.n
    e_t_size = 256
    context_size = 256
    nb_steps_warmup = int(8e4)
    nb_steps = int(1e6)
    buffer_size = 5e4
    learning_rate = 1e-4
    target_model_update = 0.999
    clipnorm = 5.
    switch_rate = 200
    window_length = 12
    batch_size = 32
    #Callbacks
    log = TrainEpisodeLogger()
    callbacks = [log]

    #Initialize our MQN model.
    
    model = MQNmodel(e_t_size, context_size, batch_size, window_length, nb_actions)
    target_model = MQNmodel(e_t_size, context_size, batch_size, window_length, nb_actions)
    target_model.set_weights(model.get_weights())
    
    #DQN
    #model = DQNmodel(nb_actions, window_length, input_shape=env.maze.matrix.shape)
    #target_model = DQNmodel(nb_actions, window_length, input_shape=env.maze.matrix.shape)
    #target_model.set_weights(model.get_weights())

    #Initialize memory buffer and policy for DQN algorithm.
    experience = [SequentialMemory(limit=int(buffer_size/len(envs)), window_length=window_length)
                             for i in range(len(envs))]

    #experience = [SequentialMemory(limit=int(buffer_size), window_length=1)]

    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.,
        nb_steps=1e5
    )

    #Initialize and compile the DQN agent.
    dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update, policy=policy,
                            batch_size=batch_size)
    
    dqn.compile(RMSprop(lr=learning_rate,clipnorm=clipnorm),metrics=["mae"])

    #Load weights if weight file exists.

    if os.path.exists(weights_file):
        dqn.load_weights(weights_file)

    #Train DQN in environment.
    dqn.fit(env4, nb_steps=nb_steps, verbose=0, callbacks=callbacks)
    #history = dqn.test(env, nb_episodes=10000,visualize=False,verbose=1)

    #Save weights if weight file does not exist.
    
    if not os.path.exists(weights_file):
        dqn.save_weights(weights_file)
    
    #Visualization Tools
    showmetrics(log)

    #rewards = np.array(history.history['episode_reward'])
    #accuracy = np.mean(rewards > 0.)
    #print accuracy

    #visualizeLayer(dqn.model, dqn.layers[1], observation)
    #visualizeLayer(dqn.model, dqn.layers[2], observation)
    #visualizeLayer(dqn.model, dqn.layers[3], observation)

    return

if __name__ == "__main__":

    weights_file = None

    if len(sys.argv)  == 2:
        if(sys.argv[1].split('.')[-1] == "h5"):
            weights_file = sys.argv[1]
        else:
            print "File extension must be .h5"
            sys.exit()
    else:
        print "Incorrect number of arguments."
        print "Usage: python MQN.py [weight_filename].h5"
        sys.exit()

    main(weights_file)
    