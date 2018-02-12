from keras.layers import *
from keras import regularizers
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K

from TemporalMemory import SimpleMemory, SimpleMemoryCell

def NeuralMapModel(e_t_size, context_size, memory_size, window_length, nb_actions, maze_dim):
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
    input_layer = Input((window_length, maze_dim[0], maze_dim[1],1))
    provider = Conv3D(filters=12, kernel_size=(1,2,2), strides=(1,2,2), padding="valid")(input_layer)
    provider = Conv3D(filters=24, kernel_size=(1,2,2), strides=(1,1,1), padding="valid")(provider)

    provider = Reshape((window_length,-1))(provider)

    e = Dense(e_t_size)(provider)
    e = Dropout(rate=0.5)(e)

    '''
    # For 8 - square partial observability
    input_layer = Input((window_length,8,))

    provider = Dense(e_t_size, activation=PReLU())(input_layer)
    provider = Dropout(0.5)(provider)
    provider = Dense(e_t_size, activation=PReLU())(provider)
    provider = Dropout(0.5)(provider)

    e = Dense(e_t_size, activation=PReLU())(provider)
    e = Dropout(0.5)(e)
    '''
    context = Dense(context_size, activation="linear")(e)
    context = Dropout(0.5)(context)

    conc = Concatenate()([e, context])

    memory = SimpleMemory(context_size, memory_size=memory_size, return_sequences=True)(conc)
    output_layer = Dense(context_size, activation=PReLU())(context)
    output_layer = Dropout(0.5)(output_layer)
    output_layer = Add()([output_layer, memory])
    output_layer = Dense(context_size, activation=PReLU())(output_layer)
    output_layer = Dropout(0.5)(output_layer)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(nb_actions, activation="linear")(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    print model.summary()
    #plot_model(model, to_file='mqn_model.png')
    return model
