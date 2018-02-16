from __future__ import absolute_import
import numpy as np
import functools
import warnings

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.layers.recurrent import RNN

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces
import tensorflow as tf

class NeuralMap(Layer):
    """Neural Map Implementation.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        memory_size: Number of observations stored in the memory.
            Set equal to the [h,w].
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 memory_size=(10,10),
                 **kwargs):

        super(NeuralMap, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.memory_size = memory_size

    def build(self, input_shape):
        input_dim = input_shape[0][-1]

        self.neural_map_shape = (self.memory_size[0], self.memory_size[1], self.units)
        self.neural_map = tf.Variable(
            initial_value=np.random.uniform(size=(1,)+self.neural_map_shape),
            dtype='float32',
            trainable=False,
            name="Neural_Map"
        )
        '''
        # kernels for Deep CNN for global read (r_t)
        kernel1_shape = (3, 3, self.units, 8)
        self.conv_kernel1 = self.add_weight(shape=kernel1_shape,
                                            name='conv_kernel1',
                                            #trainable=False,
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        kernel2_shape = (3, 3, 8, 16)
        self.conv_kernel2 = self.add_weight(shape=kernel2_shape,
                                            name='conv_kernel2',
                                            #trainable=False,
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        dense1_shape = ((16 * (self.memory_size[0] - 3) * (self.memory_size[1] - 3)), 32)
        self.conv_dense1 = self.add_weight(shape=dense1_shape,
                                           name='conv_dense1',
                                           #trainable=False,
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        dense2_shape = (32, self.units)
        self.conv_dense2 = self.add_weight(shape=dense2_shape,
                                           name='conv_dense2',
                                           #trainable=False,
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)

        # kernels for context read operation (c_t)
'''
        ckernel_shape = ((input_dim + self.units), self.units)
        self.context_kernel = self.add_weight(shape=ckernel_shape,
                                              name='context_kernel',
                                              #trainable=False,
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        # kernels for writing new vector into memory

        kernel_shape = (self.units, self.units)
        self.write_kernel = self.add_weight(shape=kernel_shape,
                                             name='write_kernel',
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                            trainable=False,
                                            constraint=self.kernel_constraint)
        self.write_update = self.add_weight(shape=kernel_shape,
                                             name='update_kernel',
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                            trainable=False,
                                            constraint=self.kernel_constraint)

        #self.write_kernel = self.recurr_kernel[:input_dim, :self.units]
        #self.write_update = self.recurr_kernel[input_dim:, :self.units]

        self.built = True

    def call(self, inputs, **kwargs):

        context = inputs[0]
        '''
        x, y = (inputs[1][:, 0], inputs[1][:, 1])
        # preprocessing states to get memory


        # Need this for later computations
        batch_size = tf.shape(context)[0]

        # global read operation
        # r_t = read(M_t) ; output dimension of r_t = self.units
        first_conv = K.conv2d(self.neural_map, self.conv_kernel1, strides=(1, 1), padding='same')
        second_conv = K.conv2d(first_conv, self.conv_kernel2, strides=(1, 1), padding='valid')
        pool_conv = K.pool2d(second_conv, pool_size=(2, 2), strides=(1, 1), padding='valid', pool_mode='avg')
        flatten_conv = K.reshape(pool_conv, (1, -1))
        dense1_conv = K.dot(flatten_conv, self.conv_dense1)
        dense2_conv = K.dot(dense1_conv, self.conv_dense2)
        r_t = dense2_conv
        r_t = K.tile(r_t, K.stack([batch_size, 1]))


        # context read operation
        # c_t = context(M_t, s_t, r_t)
        q_t = K.concatenate([context, r_t])  # [1x(s+c)]
        q_t = K.dot(q_t, self.context_kernel)  # [1xc]
        # at = K.exp(K.dot(q_t, memory)) # [1xhxw]
        # at_sum = K.sum(at)
        # at_sum_repeated = K.repeat (at_sum, (self.memory_size[0], self.memory_size[1]))
        # at /= at_sum # [1xhxw]

        # Compute attention with softmax, use batch_dot function
        # Keras is garbage, and so am I
        q_t_repeated = K.reshape(q_t, (-1, 1, 1, self.units))
        q_t_repeated = K.repeat_elements(q_t_repeated, self.memory_size[0], axis=1)
        q_t_repeated = K.repeat_elements(q_t_repeated, self.memory_size[1], axis=2)

        # Annoying conversion
        neural_map_tensor = tf.convert_to_tensor(self.neural_map)

        at = K.sum(q_t_repeated * neural_map_tensor, axis=3)
        at_sum = K.sum(at, axis=1, keepdims=True)
        at_sum = K.sum(at_sum, axis=2, keepdims=True)
        at_sum = K.repeat_elements(at_sum, self.memory_size[0], 1)
        at_sum = K.repeat_elements(at_sum, self.memory_size[1], 2)

        at = at / at_sum
        at = K.expand_dims(at, axis=3)
        at = K.repeat_elements(at, self.units, 3)

        c_t = K.sum(at * neural_map_tensor, axis=1)
        c_t = K.sum(c_t, axis=1)

        # Computing write value
        # m_t+1_x,y = write(s_t, r_t, c_t, m_t_x,y)
        s_t = K.dot(context, self.write_kernel)  # [1xc]
        global_imp = K.batch_dot(s_t, r_t, axes=1)  # [1x1]
        local_imp = K.batch_dot(s_t, c_t, axes=1)  # [1x1]

        # Convert batched x,y coordinates into indices for memory
        # We want to index into memory by doing something like
        # memory[sample_number, y, x] so that we get a {units} dimensional
        # vector out. When we do this for every sample in the batch, we'll
        # end up with a {number_batches, units} dimensional matrix.

        # First we copy and reshape the memory into {batch_size x H x W x units}
        meme = K.zeros_like(y)
        idx = K.stack([meme, y, x], axis=1)  # {batch_size x 3}

        # And index into the memory using tf.gather_nd
        mem_t = tf.gather_nd(neural_map_tensor, idx)  # {batch_size x units}
        mem_t += (local_imp / (local_imp + global_imp)) * K.dot((mem_t - s_t), self.write_update)

        # We have to reshape the memory b/c of how we calculated indices:

        # Would love to just use this, but tf has a stick in its butt
        self.neural_map = tf.scatter_nd_update(self.neural_map, idx, mem_t)

        # memory[:, :] = mem_t
        '''
        return context

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'memory_size': self.memory_size
                  }
        base_config = super(NeuralMap, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))