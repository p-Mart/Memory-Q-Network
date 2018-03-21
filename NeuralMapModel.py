from keras.layers import *
from keras import regularizers
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K

from SpatialMemory2 import *

def NeuralMapModel(e_t_size, context_size, window_length, nb_actions, maze_dim, memory_size):
    '''
    Architecture of the MQN.
    Initialize by calling:
    model = MQNmodel(e_t_size, context_size)
    where e_t_size is the dimension of the
    encoding of the convolutional layer, and
    context_size is the dimension of the 
    context layer output.
    nb_actions is the number of actions
    in the environment.
    '''

    #This is for the maze environment with partial observability
    input_layer = Input((maze_dim[0], maze_dim[1],1))

    provider = Flatten()(input_layer)
    e = Dense(e_t_size, activation="relu")(provider)

    context = Dense(context_size, activation="relu")(e)

    xy_pos = Input((2,), dtype='int32')
    memory = NeuralMap(context_size, memory_size=memory_size)([context, xy_pos])

    output_layer = Dense(context_size, activation='relu')(context)
    output_layer = Add()([output_layer, memory])
    output_layer = Dropout(0.7)(output_layer)
    output_layer = Dense(context_size, activation='relu')(output_layer)
    output_layer = Dropout(0.7)(output_layer)

    output_layer = Dense(nb_actions, activation="linear")(output_layer)

    model = Model(inputs=[input_layer, xy_pos], outputs=output_layer)
    print model.summary()
    return model
