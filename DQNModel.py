from keras.layers import *
from keras.models import Model

def DQNmodel(nb_actions, window_length, h_size, maze_dim):
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

    input_layer = Input((window_length, 4))
    provider = Conv1D(filters=12, kernel_size=1, strides=1, padding="valid")(input_layer)
    provider = Conv1D(filters=24, kernel_size=1, strides=1, padding="valid")(provider)
    #e = Flatten()(input_layer)
    #e = Dense(512)(e)
    #provider = Flatten()(provider)

    provider = Reshape((window_length,-1))(provider)

    e = Dense(h_size, activation=PReLU())(provider)
    e = Dropout(0.5)(e)
    e = Dense(h_size, activation=PReLU())(e)
    e = Dropout(0.5)(e)
    e = Flatten()(e)
    e = Dense(nb_actions, activation='linear')(e)
    e = Dropout(0.5)(e)

    model = Model(inputs=input_layer, outputs=e)
    print (model.summary())
    return model


def DistributionalDQNmodel(nb_actions, window_length, nb_atoms):
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


    #provider = Conv3D(filters=12, kernel_size=(1,2,2), strides=(1,2,2), padding="valid")(input_layer)
    #provider = Conv3D(filters=24, kernel_size=(1,2,2), strides=(1,1,1), padding="valid")(provider)
    #e = Flatten()(input_layer)
    #e = Dense(512)(e)
    #provider = Flatten()(provider)

    input_layer = Input((window_length,))
    provider = Reshape((window_length,-1))(input_layer)

    e = Dense(512)(provider)
    e = Dropout(rate=0.5)(e)
    e = Dense(512)(e)
    e = Dropout(rate=0.5)(e)
    e = Dense(nb_actions, activation='linear')(e)

    outputs = []

    for i in range(nb_actions):
        outputs.append(Dense(nb_atoms, activation="softmax")(output_layer))


    model = Model(inputs=input_layer, outputs=outputs)
    return model
