import sys, os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.callbacks import TrainEpisodeLogger

import gym
from openai_maze_envs import gym_maze
from MQNModel import MQNmodel

#Need to update environment to have indicator tiles
#Need better metrics (percent of episodes)
#Need model saving / loading
#Need model visualization ability

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

    #Running average
    #episodic_reward_means = (np.cumsum(episodic_reward_means)  
                                                #/ (np.arange(len(episodic_reward_means)) + 1))

    plt.figure(1)
    plt.subplot(311)
    plt.title("Loss per Episode")
    plt.plot(episodic_losses, 'b')

    plt.subplot(312)
    plt.title("Mean Q Value per Episode")
    plt.plot(episodic_mean_q, 'r')

    plt.subplot(313)
    plt.title("Reward")
    plt.plot(episodic_reward_means, 'g')
    plt.show()

def main(weights_file):
    #Initialize maze environments.
    env = gym.make('MazeF4-v0')
    env2 = gym.make('MazeF1-v0')
    env3 = gym.make('MazeF2-v0')
    env4 = gym.make('MazeF3-v0')
    env5 = gym.make('Maze5-v0')
    env6 = gym.make('BMaze4-v0')

    envs = [env,env2,env3,env4,env5,env6]
    
    #Setting hyperparameters.
    nb_actions = env.action_space.n
    e_t_size = 16
    context_size = 16
    nb_steps_warmup = int(2e4)
    nb_steps = int(1e5)
    buffer_size = 6e6
    learning_rate = 1e-4
    target_model_update = 0.999
    clipnorm = 20.
    switch_rate = 1000

    #Callbacks
    log = TrainEpisodeLogger()
    callbacks = [log]

    #Initialize our model.
    model = MQNmodel(e_t_size, context_size, nb_actions)
    target_model = MQNmodel(e_t_size, context_size, nb_actions)
    target_model.set_weights(model.get_weights())

    #Initialize memory buffer and policy for DQN algorithm.
    experience = [SequentialMemory(limit=int(buffer_size/len(envs)), window_length=1) for i in range(len(envs))]

    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.,
        nb_steps=nb_steps_warmup
    )

    #Initialize and compile the DQN agent.
    dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update, policy=policy)
    
    dqn.compile(RMSprop(lr=learning_rate,clipnorm=clipnorm),metrics=["mae"])

    #Load weights if weight file exists.
    if os.path.exists(weights_file):
        dqn.load_weights(weights_file)

    #Train DQN in environment.
    dqn.fit(env, envs=envs, switch_rate=switch_rate, nb_steps=nb_steps, visualize=True, verbose=0, callbacks=callbacks)

    #Save weights if weight file does not exist.
    if not os.path.exists(weights_file):
        dqn.save_weights(weights_file)

    #Visualization Tools
    showmetrics(log)

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