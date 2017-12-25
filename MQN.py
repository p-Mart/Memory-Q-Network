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

#Need to update environment to have indicator tiles
#Need better metrics (percent of episodes)
#Need model saving / loading
#Need model visualization ability
class TaxiProcessor(Processor):
    def process_observation(self, observation):
        processed_observation = observation / 500.
        return processed_observation

    def process_state_batch(self, batch):
        processed_batch = batch / 500.
        return processed_batch

def outputLayer(model, sample):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

    # Testing
    layer_outs = functor([sample, 1.])

    for idx, layer in enumerate(layer_outs):
        print "\nLayer {}:".format(idx)
        print "Shape: ", layer.shape
        print layer

    #print layer_outs

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

def logmetrics(log, model_name):

    ##### Metrics #####
    episode_rewards = []
    episodic_reward_means = []
    episodic_losses = []
    episodic_mean_q = []

    #Iterate through log for rewards.
    for key, value in log.rewards.items():
        episode_rewards.append(np.sum(log.rewards[key]))
        episodic_reward_means.append(np.mean(log.rewards[key]))

    #Iterate through log for other metrics.
    for key, value in log.episodic_metrics_variables.items():
        for i in range(0, len(log.episodic_metrics_variables[key]), 2):
            name = log.episodic_metrics_variables[key][i]
            val = log.episodic_metrics_variables[key][i+1]
            if(name == "loss" and val != '--'):
                episodic_losses.append(val)
            if(name == "mean_q" and val != '--'):
                episodic_mean_q.append(val)


    #Episodes represented per tick on plot.
    window_length = 2
    nb_episodes = len(episode_rewards)

    #Partition episodes by window_length
    nb_partitions = nb_episodes // window_length
    episode_rewards = np.array(episode_rewards[:nb_partitions * window_length])
    episode_rewards = episode_rewards.reshape((window_length, nb_partitions))

    #Save rewards for future use.
    filename = "data/{}/{}_episodes-rewards.npy".format(model_name, nb_episodes)

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


    np.save(filename, episode_rewards)

    #Plot rewards.
    sns.tsplot(data=episode_rewards)

    plt.savefig("data/{}/{}_episodes-rewards.png".format(model_name, nb_episodes))

def main(weights_file, options):
    #Initialize maze environments.
    #env = gym.make('IMaze3-v0')
    env = gym.make('Taxi-v2')

    envs = [env]
    #Setting hyperparameters.
    nb_actions = env.action_space.n
    obs_dimensions = env.observation_space.n
    e_t_size = 12
    context_size = 12
    nb_steps_warmup = int(5e3)
    nb_steps = int(4e6)
    buffer_size = 5e4
    learning_rate = 1e-3
    target_model_update = 0.999
    clipnorm = 1.
    switch_rate = 50
    window_length = 12

    #Callbacks
    log = TrainEpisodeLogger()
    callbacks = [log]

    #MQN model.
    
    #model = MQNmodel(e_t_size, context_size, window_length, nb_actions)
    #target_model = MQNmodel(e_t_size, context_size, window_length, nb_actions)


    #Distributional MQN model.
    nb_atoms = 6
    v_min = -9.5
    v_max = 20.5
    model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)
    target_model = DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions)
    target_model.set_weights(model.get_weights())
    
    #DQN model
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
        nb_steps=7e4
    )

    processor = TaxiProcessor()

    #Initialize and compile the DQN agent.
    '''
    dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update, policy=policy,
                            batch_size=32)
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
        processor=processor,
        batch_size=32
    )

    dqn.compile(Adam(lr=learning_rate,clipnorm=clipnorm),metrics=["mae"])
    #observation = env.reset()
    #print env.maze.matrix.shape
    #observation = np.reshape(observation, ((1,1,) + env.maze.matrix.shape))
    #print observation
    #print dqn.layers[1]
    #Load weights if weight file exists.
    


    #Weights will be loaded if weight file exists.
    if os.path.exists(weights_file):
        dqn.load_weights(weights_file)

    #Train DQN in environment.
    if "train" in options:
        dqn.fit(env, nb_steps=nb_steps, verbose=0, callbacks=callbacks)
    if "test" in options:
        dqn.test(env, nb_episodes=100,visualize=True)


    #Save weights.
    dqn.save_weights(weights_file)
    
    #Visualization / Logging Tools
    model_name = weights_file.split(".")[0]
    logmetrics(log, model_name)

    #Debugging
    if "debug" in options:
        observation = env.reset()
        outputLayer(dqn.model, np.array(experience[0].sample(32)[0].state0).reshape((1,12)))
        #visualizeLayer(dqn.model, dqn.layers[1], observation)
            

    return

if __name__ == "__main__":

    weights_file = None
    options = []

    if len(sys.argv)  >= 3:
        if(sys.argv[1].split('.')[-1] == "h5"):
            weights_file = sys.argv[1]
            options = sys.argv[2:]
        else:
            print "File extension must be .h5"
            sys.exit()
    else:
        print "Incorrect number of arguments."
        print "Usage: python MQN.py [weight_filename].h5 [train | test]"
        sys.exit()

    main(weights_file, options)
    