import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Model

from keras.optimizers import Adam
from TemporalMemory import SimpleMemory, SimpleMemoryCell
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainEpisodeLogger

import gym
from openai_maze_envs import gym_maze

env = gym.make('MazeF1-v1')
nb_actions = env.action_space.n

print nb_actions
print env.observation_space

e_t_size = 16
context_size = 16


def MQNmodel(e_t_size, context_size):

    input_layer = Input((None,) + (8,))

    provider = Dense(64)(input_layer)
    provider = Dense(64)(provider)
    provider = Dense(64)(provider)
    provider = Dense(64)(provider)

    e = Dense(e_t_size)(provider)
    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])

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



model = MQNmodel(e_t_size, context_size)
target_model = MQNmodel(e_t_size, context_size)
target_model.set_weights(model.get_weights())

experience = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

log = TrainEpisodeLogger()
callbacks = [log]

dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3),metrics=["mae"])

dqn.fit(env, nb_steps=100000, visualize=False, verbose=0, callbacks=callbacks)


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

