from keras.layers import *
from keras import regularizers
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K

from TemporalMemory import SimpleMemory, SimpleMemoryCell

def MQNmodel(e_t_size, context_size, batch_size, window_length, nb_actions, obs_dimensions):
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
    '''
    #This is for the maze environment.
    input_layer = Input((window_length,7, 12,1))
    provider = Conv3D(filters=12, kernel_size=(1,2,2), strides=(1,2,2), padding="valid")(input_layer)
    provider = Conv3D(filters=24, kernel_size=(1,2,2), strides=(1,1,1), padding="valid")(provider)

    provider = Reshape((window_length,-1))(provider)

    e = Dense(e_t_size)(provider)
    e = Dropout(rate=0.5)(e)
    '''
    input_layer = Input((window_length,))
    provider = Reshape((window_length,-1))(input_layer)

    e = Dense(e_t_size)(provider)
    e = Dropout(rate=0.5)(e)

    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])

    memory = SimpleMemory(context_size, memory_size=24, return_sequences=True)(conc)
    output_layer = Dense(context_size, activation="linear")(context)
    #output_layer = Reshape((context_size,))(output_layer)
    output_layer = Lambda(lambda x: K.relu(x[0] + x[1]))([output_layer, memory])
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Flatten()(output_layer)
    #output_layer = Reshape((batch_size*context_size,))(output_layer)
    output_layer = Dense(nb_actions, activation="linear")(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    print model.summary()

    return model

def DistributionalMQNModel(e_t_size, context_size, window_length, nb_actions, nb_atoms, obs_dimensions):
    input_layer = Input((window_length,7, 12,1))
    provider = Conv3D(filters=12, kernel_size=(1,2,2), strides=(1,2,2), padding="valid")(input_layer)
    provider = Conv3D(filters=24, kernel_size=(1,2,2), strides=(1,1,1), padding="valid")(provider)

    provider = Reshape((window_length,-1))(provider)

    #input_layer = Input((window_length,))
    #provider = Reshape((window_length,-1))(input_layer)

    e = Dense(e_t_size)(provider)
    e = Dropout(rate=0.5)(e)

    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])

    memory = SimpleMemory(
        context_size, 
        memory_size=12, 
        return_sequences=True
    )(conc)

    output_layer = Dense(context_size, activation="linear")(context)
    #output_layer = Reshape((context_size,))(output_layer)
    output_layer = Lambda(lambda x: K.relu(x[0] + x[1]))([output_layer, memory])
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Flatten()(output_layer)

    outputs = []

    for i in range(nb_actions):
        outputs.append(Dense(nb_atoms, activation="softmax")(output_layer))


    model = Model(inputs=input_layer, outputs=outputs)
    print model.summary()
    #plot_model(model, to_file='distmodel.png')
    return model