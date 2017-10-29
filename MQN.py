import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import Callback

from keras.optimizers import RMSprop
from TemporalMemory import SimpleMemory, SimpleMemoryCell
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, Memory
from rl.callbacks import TrainEpisodeLogger

import gym
from openai_maze_envs import gym_maze

env = gym.make('MazeF4-v0')
env2 = gym.make('MazeF1-v0')
env3 = gym.make('MazeF2-v0')
env4 = gym.make('MazeF3-v0')
env5 = gym.make('Maze5-v0')
env6 = gym.make('BMaze4-v0')

envs = [env,env2,env3,env4,env5,env6]

nb_actions = env.action_space.n

e_t_size = 16
context_size = 16

'''
class EnvDependentMemory(Memory):

    def __init__(self, limit, envs,**kwargs):
        super(EnvDependentMemory, self).__init__(**kwargs)

        self.context = {}


    def sample(self, batch_size, batch_idxs=None):
'''



class RandomizeEnvironmentOnEpisode(Callback):
    def __init__(self, done):
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.model.stop_training=True
        done[0] = True

def MQNmodel(e_t_size, context_size):

    input_layer = Input((1,None,None))

    provider = Conv2D(filters=24, kernel_size=(2,2), padding="same", data_format="channels_first")(input_layer)
    provider = Conv2D(filters=12, kernel_size=(2,2), padding="same", data_format="channels_first")(provider)
    print provider
    '''
    provider = Dense(64)(input_layer)
    provider = Dense(64)(provider)
    provider = Dense(64)(provider)
    provider = Dense(64)(provider)
    '''
    #provider = Reshape((-1,))(provider)
    provider = GlobalMaxPooling2D(data_format="channels_first")(provider)
    print provider
    e = Dense(e_t_size)(provider)
    #e = Flatten()(e)
    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])
    conc = Reshape((1, -1))(conc)
    memory = SimpleMemory(context_size, memory_size=11)(conc)

    #Need to add the correct output as per paper
    output_layer = Dense(context_size, activation="linear")(context)
    #output_layer = Dense(context_size)(context)
    output_layer = Reshape((context_size,))(output_layer)
    output_layer = Lambda(lambda x: K.relu(x[0] + x[1]))([output_layer, memory])
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Dense(nb_actions, activation="linear")(output_layer)


    model = Model(inputs=input_layer, outputs=output_layer)
    print model.summary()
    return model

nb_steps_warmup = int(1e5)
nb_steps = int(1e6)

model = MQNmodel(e_t_size, context_size)
target_model = MQNmodel(e_t_size, context_size)
target_model.set_weights(model.get_weights())

buffer_size = 6e6
experience = [SequentialMemory(limit=int(buffer_size/len(envs)), window_length=1) for i in range(len(envs))]

policy = LinearAnnealedPolicy(
    inner_policy=EpsGreedyQPolicy(),
    attr="eps",
    value_max=1.0,
    value_min=0.1,
    value_test=0.,
    nb_steps=nb_steps_warmup
)

log = TrainEpisodeLogger()
callbacks = [log]

dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=nb_steps_warmup, target_model_update=0.999, policy=policy)
dqn.compile(RMSprop(lr=1e-4,clipnorm=20.),metrics=["mae"])
dqn.fit(env, envs=envs, nb_steps=nb_steps, verbose=0, callbacks=callbacks)
'''
for i in range(nb_episodes):

    print "Episode {}:".format(i)
    dqn.fit(env, nb_steps=max_steps, visualize=False, verbose=0, callbacks=callbacks)
    print len(log.observations)
    #if(dqn.nb_steps_warmup > )

    if(done[0] == True):
        done[0] = False
        env = np.random.choice(envs)
'''
dqn.save_weights('dqn_weights_singlegoal_v2.h5')

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

#Need to update environment to have indicator tiles
#Need better metrics (percent of episodes)
#Need model saving / loading
#Need model visualization ability