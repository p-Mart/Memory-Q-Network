from keras.layers import *
from keras.models import Model

from TemporalMemory import SimpleMemory, SimpleMemoryCell

def MQNmodel(e_t_size, context_size, nb_actions):
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
    input_layer = Input((1,None,None))

    provider = Conv2D(filters=24, kernel_size=(2,2), padding="same", data_format="channels_first")(input_layer)
    provider = Conv2D(filters=12, kernel_size=(2,2), padding="same", data_format="channels_first")(provider)
    provider = GlobalMaxPooling2D(data_format="channels_first")(provider)

    e = Dense(e_t_size)(provider)
    context = Dense(context_size, activation="linear")(e)

    conc = Concatenate()([e, context])
    conc = Reshape((1, -1))(conc)
    memory = SimpleMemory(context_size, memory_size=11)(conc)

    output_layer = Dense(context_size, activation="linear")(context)
    output_layer = Reshape((context_size,))(output_layer)
    output_layer = Lambda(lambda x: K.relu(x[0] + x[1]))([output_layer, memory])
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Dense(nb_actions, activation="linear")(output_layer)


    model = Model(inputs=input_layer, outputs=output_layer)
    return model