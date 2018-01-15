import sys, os, errno

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from rl.core import Processor
import keras.backend as K

class TaxiProcessor(Processor):
    '''Processor for the taxi environment.
    Normalizes the input values to be within
    the range (0, 1).'''
    def process_observation(self, observation):
        processed_observation = observation / 500.
        return processed_observation

    def process_state_batch(self, batch):
        processed_batch = batch / 500.
        return processed_batch

def outputLayer(model, sample):
    '''A function allowing us to get the
    raw output of every layer in a model,
    given some sample as input.'''
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
    '''This only visualizes the output of a
    CNN layer at the moment. Takes
    your model as input, the layer number
    of the Convolutional layer, and an
    input sample to test with.'''

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


def logHyperparameters(model_name, **kwargs):
    '''Takes an arbitrary number of named arguments
    as input, and logs their name and value. Use this
    to log hyperparameters in model'''

    filename = "data/{}/hyperparams.txt".format(model_name)

    with open(filename, 'w') as f:
        for key in kwargs:
            f.write("{} : {}\n".format(key, kwargs[key]))



def logmetrics(log, model_name, window_length=50):
    '''Utility function to log performance of model.
    Takes a callback log, the name of your model,
    and an optional number of episodes represented 
    per tick on the plot of model performance 
    (window_length).'''

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


    #Partition episodes by window_length
    nb_episodes = len(episode_rewards)
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
    plt.title("Avg. Reward for {} episode intervals".format(window_length))
    plt.ylabel("Reward")
    plt.xlabel("1 tick per {} episodes".format(window_length))
    plt.savefig("data/{}/{}_episodes-rewards.png".format(model_name, nb_episodes))