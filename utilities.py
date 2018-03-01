import sys, os, errno

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from rl.core import Processor
from rl.callbacks import Callback

import keras.backend as K
import tensorflow as tf

class RLTensorBoard(Callback):
    def __init__(self, log_dir='./logs', histogram_freq=20):
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()

        # Log all layer outputs for each layer
        #layer_names, layer_outputs = outputLayer(self.model, self.model.get_sample())
        #tf.summary.histogram(layer_names, layer_outputs)
        self.writer = tf.summary.FileWriter(self.log_dir,
                                            self.sess.graph)

    def log_histogram(self, name, values, episode, bins=1000):
        """
        Creates a histogram object for tensorboard
        (https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41)
        """

        values = np.array(values)
        # values = values.eval(session=self.sess)
        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    def on_episode_end(self, episode, logs=None):
        logs = logs or {}

        # Log the weights and gradients to Tensorboard every histogram_freq steps.
        # Getting the gradients takes a long time, so this should only be used
        # for debugging.
        if episode % self.histogram_freq == 0:
            for layer in self.model.model.layers:
                weights = [weight for weight in layer.weights if layer.trainable]
                for weight in weights:
                    self.log_histogram(weight.name, weight.eval(session=self.sess), episode)

                '''TODO : Get gradients
                gradients = model.optimizer.get_gradients(model.total_loss, weights)
                input_tensors = [model.inputs[0], model.sample_weights[0], model.targets[0], K.learning_phase()]
                get_gradients = K.function(inputs=input_tensors, outputs=gradients)

                inputs = [
                    model.inputs[0].eval(session=self.sess),
                    model.sample_weights[0].eval(session=self.sess),
                    model.targets[0].eval(session=self.sess),
                    1
                ]

                self.log_histogram(gradients.name, get_gradients(inputs), episode)
                '''

        # Log scalar values to TensorBoard contained in log variable (reward, loss, etc.)
        for name, value in logs.items():

            summary = tf.Summary()

            # Metrics is a list of misc. metrics returned from the backprop step.
            if name == 'metrics':
                for idx, metric in enumerate(value):
                    if metric != np.nan:
                        summary_value = summary.value.add()
                        summary_value.simple_value = metric
                        summary_value.tag = self.model.metrics_names[idx]
                        self.writer.add_summary(summary, episode)

                continue



            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, episode)

        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


class PongProcessor(Processor):
    def process_observation(self, observation):
        #reduce game frame to just paddle and ball
        observation = observation[35:195] # cropping
        observation = observation[::2, ::2, 0]
        observation[observation == 144] = 0 #erase background
        observation[observation == 109] = 0
        observation[observation != 0] = 1 #paddles and balls set to 1
        observation = observation.astype(np.float).ravel()
        # print(observation.shape)
        # np.squeeze(x, axis=(2,)).shape
        # return np.squeeze(observation, axis=(2,))
        return observation#.reshape((1,6400)) #flatten to 1d array

    def process_state_batch(self, batch):
        # #reduce game frame to just paddle and ball
        # batch = batch[35:195] # cropping
        # batch = batch[::2, ::2, 0]
        # batch[batch == 144] = 0 #erase background
        # batch[batch == 109] = 0
        # batch[batch != 0] = 1 #paddles and balls set to 1
        # print(batch[2].shape)
        # batch = batch.astype(np.float).ravel()
        return batch
        # return batch.reshape((6400,1))
        # return np.squeeze(batch, axis=(2,))
        # return np.reshape(batch,(1,6400)) #flatten to 1d array



class TaxiProcessor(Processor):
    """Processor for the taxi environment.
    Normalizes the input values to be within
    the range (0, 1)."""
    def process_observation(self, observation):
        processed_observation = observation / 500.
        return processed_observation

    def process_state_batch(self, batch):
        processed_batch = batch / 500.
        return processed_batch


class MazeProcessor(Processor):
    """Processor for the maze environment.
    Normalizes the input values to be within
    the range (0, 1)."""
    def process_observation(self, observation):
        processed_observation = np.array(observation, dtype=np.int8) / 10.
        return processed_observation

    def process_state_batch(self, batch):
        processed_batch = np.array(batch, dtype=np.int8) / 10.
        return processed_batch


def outputWeights(model):
    """
    Gets the weight of each layer in model. Meant for easy logging
    in tensorboard.

    Arguments:
        model : A Keras model

    returns : map of each layer to their weights
    """
    weights = [(layer.get_weights.name, layer.get_weights()) for layer in model.layers]

    return zip(model.layers, weights)


def outputLayer(model, sample):
    """
    A function allowing us to get the
    raw activation of every layer in a model,
    given some sample as input.

    Arguments:
        model : A Keras model
        sample : An sample to be passed in as input to model

    returns : map of each layer name to their output for a given sample
    """
    inp = model.input                                           # input placeholder
    activations = [layer.output for layer in model.layers]          # all layer outputs
    activation_functor = K.function([inp]+ [K.learning_phase()], activations ) # evaluation function

    # Testing
    layer_activations = activation_functor([sample, 1.])

    # Map layer outputs to corresponding layer names.
    return zip([layer.name for layer in model.layers], layer_outs)


def visualizeLayer(model, layer, sample):
    """This only visualizes the output of a
    CNN layer at the moment. Takes
    your model as input, the layer number
    of the Convolutional layer, and an
    input sample to test with."""

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
    """Takes an arbitrary number of named arguments
    as input, and logs their name and value. Use this
    to log hyperparameters in model"""

    filename = "data/{}/hyperparams.txt".format(model_name)

    with open(filename, 'w') as f:
        for key in kwargs:
            f.write("{} : {}\n".format(key, kwargs[key]))


def logmetrics(log, model_name, window_length=50):
    """Utility function to log performance of model.
    Takes a callback log, the name of your model,
    and an optional number of episodes represented
    per tick on the plot of model performance
    (window_length)."""

    episode_rewards = []
    episode_losses = []
    episodic_mean_q = []

    # Iterate through log for rewards.
    for key, value in log.rewards.items():
        episode_rewards.append(np.sum(log.rewards[key]))

    # Iterate through log for other metrics.
    for key, value in log.episodic_metrics_variables.items():
        for i in range(0, len(log.episodic_metrics_variables[key]), 2):
            name = log.episodic_metrics_variables[key][i]
            val = log.episodic_metrics_variables[key][i+1]
            if(name == "loss" and val != '--'):
                episode_losses.append(val)
            if(name == "mean_q" and val != '--'):
                episodic_mean_q.append(val)

    # Partition episodes by window_length
    nb_episodes = len(episode_losses)
    nb_partitions = nb_episodes // window_length

    # Compute Metrics with error bars
    episode_rewards = np.array(episode_rewards[:nb_partitions * window_length])
    episode_rewards = episode_rewards.reshape((window_length, nb_partitions))

    nb_losses = len(episode_losses)
    nb_partitions = nb_losses // window_length
    episode_losses  = np.array(episode_losses[:nb_partitions* window_length])
    episode_losses = episode_losses.reshape((window_length, nb_partitions))

    # Save metrics for future use.
    rewards_filename = "data/{}/{}_episodes-rewards.npy".format(model_name, nb_episodes)
    losses_filename = "data/{}/{}_episodes-losses.npy".format(model_name, nb_episodes)

    if not os.path.exists(os.path.dirname(rewards_filename)):
        try:
            os.makedirs(os.path.dirname(rewards_filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    np.save(rewards_filename, episode_rewards)
    np.save(losses_filename, episode_losses)

    # Plot rewards.
    plt.figure(1)
    sns.tsplot(data=episode_rewards)
    plt.title("Avg. Reward for {} episode intervals".format(window_length))
    plt.ylabel("Reward")
    plt.xlabel("1 tick per {} episodes".format(window_length))
    plt.savefig("data/{}/{}_episodes-rewards.png".format(model_name, nb_episodes))

    plt.figure(2)
    sns.tsplot(data=episode_losses)
    plt.title("Avg. Loss for {} episode intervals".format(window_length))
    plt.ylabel("Loss")
    plt.xlabel("1 tick per {} episodes".format(window_length))
    plt.savefig("data/{}/{}_episodes-losses.png".format(model_name, nb_episodes))
