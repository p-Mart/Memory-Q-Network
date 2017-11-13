from keras.layers import *
from keras.models import Model

def DQNmodel(nb_actions, window_length, input_shape):
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
    input_layer = Input((window_length,7, 7,1))

    provider = Conv3D(filters=12, kernel_size=(1,2,2), strides=(1,2,2), padding="valid")(input_layer)
    provider = Conv3D(filters=24, kernel_size=(1,2,2), strides=(1,1,1), padding="valid")(provider)
    #e = Flatten()(input_layer)
    #e = Dense(512)(e)
    provider = Flatten()(provider)

    e = Dense(512)(provider)
    e = Dropout(rate=0.5)(e)
    e = Dense(512)(e)
    e = Dropout(rate=0.5)(e)
    e = Dense(nb_actions, activation='linear')(e)


    model = Model(inputs=input_layer, outputs=e)
    return model