import numpy as np

from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import RNN
from keras.models import Model

from keras.optimizers import Adam
from TemporalMemory import SimpleMemory, SimpleMemoryCell
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import gym

env = gym.make('CartPole-v0');
nb_actions = env.action_space.n

e_t_size = 16
context_size = 16

def MQNmodel(e_t_size, context_size):

    input_layer = Input((None,) + env.observation_space.shape)

    e = Dense(e_t_size)(input_layer)
    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])

    memory = SimpleMemory(context_size)(conc)

    output_layer = Dense(nb_actions, activation="linear")(memory)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


model = MQNmodel(e_t_size, context_size)
target_model = MQNmodel(e_t_size, context_size)
target_model.set_weights(model.get_weights())


experience = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
#model = Model(inputs=x, outputs=layer)
dqn = DQNAgent(model=model, target_model=target_model, nb_actions=nb_actions, memory=experience,
                            nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3),metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

