import sys, os, errno

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent, DistributionalDQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.callbacks import TrainEpisodeLogger

import gym
from MQNModel import *
from DQNModel import *
from utilities import *

"""
    Todo : *Add environment as an input option?
               *Automatic model loading / initialization
                   from hyperparams.txt, model_name.h5.

               *As part of the above, possibly need to contain
                   hyperparameters as part of a dictionary that
                   can be used to automate the initialization process;
                   this might even include the class name
                   (MQNModel, etc.). Would be a useful tool when
                   running large amounts of automated tests later.
"""


def main(model_name, options):

    # Initialize maze environments.
    env = gym.make('Pong-v0')
    # env = gym.make('CartPole-v0')
    #env = gym.make('Taxi-v2')

    envs = [env]

    # Setting hyperparameters.
    nb_actions = env.action_space.n
    maze_dim =  (6400,1)
    h_size = 64 # For DQN
    e_t_size = 64 #For MQN / RMQN
    context_size = 64
    nb_steps_warmup = int(1e5)
    nb_steps = int(4e5)
    buffer_size = 8e4
    learning_rate = 0.003
    target_model_update = 0.999
    clipnorm = 10.
    switch_rate = 50
    window_length = 12
    memory_size = None

    # Callbacks
    log = TrainEpisodeLogger()
    #tensorboard = TensorBoard(log_dir="./logs/{}".format(model_name))
    rl_tensorboard = RLTensorBoard(log_dir="./logs/{}".format(model_name), histogram_freq=100)

    callbacks = [log, rl_tensorboard]

    ### Models ###
    model = None
    target_model = None

    # MQN model.
    if "MQN" in options:
        memory_size = 12
        model = MQNmodel(e_t_size, context_size, memory_size, window_length, nb_actions, maze_dim)
        target_model = MQNmodel(e_t_size, context_size, memory_size, window_length, nb_actions, maze_dim)

    # RMQN model.
    if "RMQN" in options:
        memory_size = 12
        model = RMQNmodel(e_t_size, context_size, memory_size, window_length, nb_actions, maze_dim)
        target_model = RMQNmodel(e_t_size, context_size, memory_size, window_length, nb_actions, maze_dim)

    # Distributional MQN model.
    nb_atoms = 51
    v_min = -2.
    v_max = 2.
    #model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)
    #target_model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)

    # DQN model
    if "DQN" in options:
        model = DQNmodel(nb_actions, window_length, h_size, maze_dim)
        target_model = DQNmodel(nb_actions, window_length, h_size, maze_dim)

    # Initialize our target model with the same weights as our model.
    target_model.set_weights(model.get_weights())

    # Initialize memory buffer for DQN algorithm.
    experience = [SequentialMemory(limit=int(buffer_size/len(envs)), window_length=window_length)
                             for i in range(len(envs))]

    # Learning policy where we initially begin training our agent by making random moves
    # with a probability of 1, and linearly decrease that probability down to 0.1 over the
    # course of some arbitrary number of steps. (nb_steps)
    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.,
        nb_steps=1e5
    )

    # Optional processor.
    processor = PongProcessor()
    # processor = MazeProcessor()

    # Initialize and compile the DQN agent.

    dqn = DQNAgent(
        model=model,
        target_model=target_model,
        nb_actions=nb_actions,
        memory=experience,
        nb_steps_warmup=nb_steps_warmup,
        target_model_update=target_model_update,
        policy=policy,
        processor=processor,
        batch_size=8
    )


    #Initialize experimental Distributional DQN Agent
    '''
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
    '''

    # Compile the agent to check for validity, build tensorflow graph, etc.
    dqn.compile(RMSprop(lr=learning_rate, clipnorm=clipnorm), metrics=["mae"])

    # Weights will be loaded if weight file exists.
    if os.path.exists("data/{}/{}".format(model_name, model_name + ".h5")):
        dqn.load_weights("data/{}/{}".format(model_name, model_name + ".h5"))

    # Train DQN in environment.
    if "train" in options:
        dqn.fit(env, nb_steps=nb_steps, verbose=0, callbacks=callbacks)

        # Visualization / Logging Tools
        logmetrics(log, model_name)
        logHyperparameters(
            model_name,
            e_t_size=e_t_size,
            context_size=context_size,
            h_size=h_size,
            memory_size=memory_size,
            learning_rate=learning_rate,
            target_model_update=target_model_update,
            clipnorm=clipnorm,
            window_length=window_length,
            nb_atoms=nb_atoms,
            v_min=v_min,
            v_max=v_max
        )

        # Save weights.
        dqn.save_weights("data/{}/{}".format(model_name, model_name + ".h5"))

    # Test DQN in environment.
    if "test" in options:
        dqn.test(env, nb_episodes=100, visualize=True)

    #Debugging
    if "debug" in options:
        observation = env.reset()
        outputLayer(dqn.model, np.array(experience[0].sample(32)[0].state0))
        #visualizeLayer(dqn.model, dqn.layers[1], observation)

    return

if __name__ == "__main__":

    model_name = None
    options = []

    if len(sys.argv) >= 3:
        model_name = sys.argv[1]

        mode = sys.argv[2]

        if mode != "train" and mode != "test":
            print ("Usage: python MQN.py [model_name].h5 [train | test]")
            sys.exit()

        options = sys.argv[2:]

    else:
        print ("Incorrect number of arguments.")
        print ("Usage: python MQN.py [model_name].h5 [train | test]")
        sys.exit()

    main(model_name, options)
