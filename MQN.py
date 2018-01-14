import sys, os, errno

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import keras.backend as K

from rl.agents.dqn import DQNAgent, DistributionalDQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.callbacks import TrainEpisodeLogger
from rl.core import Processor

import gym
from MQNModel import MQNmodel, DistributionalMQNModel
from DQNModel import DQNmodel
from utilities import *

def main(weights_file, options):
    ###Todo : *Add environment as an input option?
    ###           *Automatic model loading / initialization
    ###               from hyperparams.txt, model_name.h5.

    #Initialize maze environments.
    env = gym.make('IMaze3-v0')
    #env = gym.make('Taxi-v2')

    envs = [env]

    #Setting hyperparameters.
    nb_actions = env.action_space.n
    obs_dimensions = env.observation_space.n
    e_t_size = 48
    context_size = 48
    nb_steps_warmup = int(5e4)
    nb_steps = int(4e6)
    buffer_size = 5e4
    learning_rate = 1e-3
    target_model_update = 0.999
    clipnorm = 10.
    switch_rate = 50
    window_length = 12

    #Callbacks
    log = TrainEpisodeLogger()
    callbacks = [log]

    #MQN model.
    
    #model = MQNmodel(e_t_size, context_size, window_length, nb_actions)
    #target_model = MQNmodel(e_t_size, context_size, window_length, nb_actions)


    #Distributional MQN model.
    nb_atoms = 51
    v_min = -2.
    v_max = 2.
    model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)
    target_model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)
    
    #DQN model
    #model = DQNmodel(nb_actions, window_length, input_shape=env.maze.matrix.shape)
    #target_model = DQNmodel(nb_actions, window_length, input_shape=env.maze.matrix.shape)

    #Initialize our target model with the same weights as our model.
    target_model.set_weights(model.get_weights())

    #Initialize memory buffer for DQN algorithm.
    experience = [SequentialMemory(limit=int(buffer_size/len(envs)), window_length=window_length)
                             for i in range(len(envs))]

    #Learning policy where we initially begin training our agent by making random moves
    #with a probability of 1, and linearly decrease that probability down to 0.1 over the
    #course of some arbitrary number of steps. (nb_steps)
    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.,
        nb_steps=7e4
    )

    #Optional processor.
    #processor = TaxiProcessor()

    #Initialize and compile the DQN agent.
    '''
    dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update, policy=policy,
                            batch_size=32)
    '''

    #Initialize experimental Distributional DQN Agent
    dqn = DistributionalDQNAgent(
        model=model, 
        target_model=target_model,
        num_atoms=nb_atoms,
        v_min=v_min,
        v_max=v_max,
        nb_actions=nb_actions, 
        memory=experience,
        nb_steps_warmup=nb_steps_warmup, 
        target_model_update=target_model_update, 
        policy=policy,
        #processor=processor,
        batch_size=32
    )

    #Compile the agent to check for validity, build tensorflow graph, etc.
    dqn.compile(Adam(lr=learning_rate,clipnorm=clipnorm),metrics=["mae"])


    #Extract model name from weights file.
    model_name = weights_file.split(".")[0]

    #Weights will be loaded if weight file exists.
    if os.path.exists("data/{}/{}".format(model_name, weights_file)):
        dqn.load_weights("data/{}/{}".format(model_name, weights_file))

    #Train DQN in environment.
    if "train" in options:
        dqn.fit(env, nb_steps=nb_steps, verbose=0, callbacks=callbacks)

        #Visualization / Logging Tools
        logmetrics(log, model_name)
        logHyperparameters(model_name,
            e_t_size=e_t_size,
            context_size=context_size,
            learning_rate=learning_rate,
            target_model_update=target_model_update,
            clipnorm=clipnorm,
            window_length=window_length,
            nb_atoms=nb_atoms,
            v_min=v_min,
            v_max=v_max)

        #Save weights.
        dqn.save_weights("data/{}/{}".format(model_name, weights_file)) 

    #Test DQN in environment.
    if "test" in options:
        dqn.test(env, nb_episodes=100,visualize=True, dump_output=True)




    #Debugging
    if "debug" in options:
        observation = env.reset()
        outputLayer(dqn.model, np.array(experience[0].sample(32)[0].state0))
        #visualizeLayer(dqn.model, dqn.layers[1], observation)
            

    return

if __name__ == "__main__":

    weights_file = None
    options = []

    if len(sys.argv)  >= 3:
        if(sys.argv[1].split('.')[-1] == "h5"):
            weights_file = sys.argv[1]
            
            mode = sys.argv[2]
            if mode != "train" or mode != "test":
                print "Usage: python MQN.py [weight_filename].h5 [train | test]"
                sys.exit()

            options = sys.argv[2:]
        else:
            print "File extension must be .h5"
            sys.exit()
    else:
        print "Incorrect number of arguments."
        print "Usage: python MQN.py [weight_filename].h5 [train | test]"
        sys.exit()

    main(weights_file, options)
    