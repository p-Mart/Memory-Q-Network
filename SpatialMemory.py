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

class NeuralMapCell(Layer):
    """Cell class for the Neural Map layer.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
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

    def __init__(self, units, pos_input,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 memory_size=[10,10],
                 **kwargs):
        super(NeuralMapCell, self).__init__(**kwargs)
        self.units = units
        self.pos_input = pos_input
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)

        self.memory_size = memory_size
        self.state_size = (self.units, ) * (1 + (self.memory_size[0] * self.memory_size[1])) # r_t + Memory [h, w]

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # kernels for Deep CNN for global read (r_t)
        kernel1_shape = (3,3,self.units,32)
        self.conv_kernel1 = self.add_weight(shape=kernel1_shape,
                                            name='conv_kernel1',
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        kernel2_shape = (3,3,32,64)
        self.conv_kernel2 = self.add_weight(shape=kernel2_shape,
                                            name='conv_kernel2',
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        dense1_shape = ((64 * (self.memory_size[0]-2) * (self.memory_size[1]-2)), 128)
        self.conv_dense1 = self.add_weight(shape=dense1_shape,
                                           name='conv_dense1',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        dense2_shape = (128, self.units)
        self.conv_dense2 = self.add_weight(shape=dense2_shape,
                                           name='conv_dense2',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)

        # kernels for context read operation (c_t)
        ckernel_shape = ((input_dim + self.units), self.units)
        self.context_kernel = self.add_weight(shape=ckernel_shape,
                                              name='context_kernel',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        # kernels for writing new vector into memory
        kernel_shape = (input_dim + self.units, self.units)
        self.recurr_kernel = self.add_weight(shape=kernel_shape,
                                        name='write_kernel',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)
        self.write_kernel = self.recurr_kernel[:input_dim, :self.units]
        self.write_update = self.recurr_kernel[input_dim:, :self.units]
        # wkernel_shape = (input_dim, self.units)
        # self.write_kernel = self.add_weight(shape=wkernel_shape,
        #                                     name='write_kernel',
        #                                     initializer=self.recurrent_initializer,
        #                                     regularizer=self.recurrent_regularizer,
        #                                     constraint=self.recurrent_constraint)
        # wukernel_shape = (self.units, self.units)
        # self.write_update = self.add_weight(shape=wukernel_shape,
        #                                     name='wukernel_shape',
        #                                     initializer=self.kernel_initializer,
        #                                     regularizer=self.kernel_regularizer,
        #                                     constraint=self.kernel_constraint)

        self.built = True

    # figure out how to send x, y to the fucntion
    def call(self, inputs, states, training=None):

        # inputs are going to come in as a tuple of size 2
        # where inputs[0] is the context prior to the neural
        # map, and inputs[1] will be another tuple - (x, y)
        context = inputs
        x, y = (self.pos_input[:, 0], self.pos_input[:, 1])
        # x, y = (K.expand_dims(x, -1), K.expand_dims(y, -1))
        # preprocessing states to get memory
        memory = states[1:((self.memory_size[0] * self.memory_size[1]) + 1)]
        memory = K.transpose(memory)
        memory = K.reshape(memory, (-1, self.units, self.memory_size[0], self.memory_size[1]))

        # Need this for later computations
        batch_size = K.shape(memory)[0]

        # global read operation
        # r_t = read(M_t) ; output dimension of r_t = self.units
        first_conv = K.conv2d(memory, self.conv_kernel1, strides=(1,1), padding='same', data_format='channels_first')
        second_conv = K.conv2d(first_conv, self.conv_kernel2, strides=(1,1), padding='valid', data_format='channels_first')
        pool_conv = K.pool2d(second_conv, pool_size=(2,2), strides=(1,1), padding='valid', data_format='channels_first', pool_mode='avg')
        flatten_conv = K.batch_flatten(pool_conv)
        dense1_conv = K.dot(flatten_conv, self.conv_dense1)
        dense2_conv = K.dot(dense1_conv, self.conv_dense2)
        r_t = dense2_conv

        # context read operation
        # c_t = context(M_t, s_t, r_t)
        q_t = K.concatenate([context, r_t]) # [1x(s+c)]
        q_t = K.dot(q_t, self.context_kernel) # [1xc]
        # at = K.exp(K.dot(q_t, memory)) # [1xhxw]
        # at_sum = K.sum(at)
        # at_sum_repeated = K.repeat (at_sum, (self.memory_size[0], self.memory_size[1]))
        # at /= at_sum # [1xhxw]

        # Compute attention with softmax, use batch_dot function
        # Keras is garbage, and so am I
        q_t_repeated = K.expand_dims(q_t, axis=2)
        q_t_repeated = K.expand_dims(q_t_repeated, axis=3)
        q_t_repeated = K.repeat_elements(q_t_repeated, self.memory_size[0], 2)
        q_t_repeated = K.repeat_elements(q_t_repeated, self.memory_size[1], 3)

        at = K.sum(q_t_repeated * memory, axis=1)
        at_sum = K.sum(at, axis=1, keepdims=True)
        at_sum = K.sum(at_sum, axis=2, keepdims=True)
        at_sum = K.repeat_elements(at_sum, self.memory_size[0], 1)
        at_sum = K.repeat_elements(at_sum, self.memory_size[1], 2)

        at = at / at_sum
        at = K.expand_dims(at, axis=1)
        at = K.repeat_elements(at, self.units, 1)

        c_t = K.sum(at * memory, axis=2)
        c_t = K.sum(c_t, axis=2)


        # Computing write value
        # m_t+1_x,y = write(s_t, r_t, c_t, m_t_x,y)
        s_t = K.dot(context, self.write_kernel) # [1xc]
        global_imp = K.batch_dot(s_t, r_t, axes=1) # [1x1]
        local_imp = K.batch_dot(s_t, c_t, axes=1) # [1x1]

        # Convert batched x,y coordinates into indices for memory
        # We want to index into memory by doing something like
        # memory[sample_number, y, x] so that we get a {units} dimensional
        # vector out. When we do this for every sample in the batch, we'll
        # end up with a {number_batches, units} dimensional matrix.

        # First we copy and reshape the memory into {batch_size x H x W x units}
        mem_t = K.reshape(memory, (batch_size, self.memory_size[0], self.memory_size[1], self.units))

        # Now we generate the indices
        B = tf.range(0, batch_size) # {batch_size}
        idx = tf.stack([B, y, x], axis=1) # {batch_size x 3}

        # And index into the memory using tf.gather_nd
        mem_t = tf.gather_nd(mem_t, idx) # {batch_size x units}
        mem_t += (local_imp / (local_imp + global_imp)) * K.dot((mem_t - s_t), self.write_update)

        # We have to reshape the memory b/c of how we calculated indices:
        memory = K.reshape(memory, (batch_size, self.memory_size[0], self.memory_size[1], self.units))

        # While loop to update memory.
        i = tf.constant(0)
        c = lambda i : tf.less(i, batch_size)
        def body(i):
            memory[i, y[i], x[i]] = mem_t[i]
            return tf.add(i, 1)

        tf.while_loop(c, body, [i])
        # Would love to just use this, but tf has a stick in its butt
        # memory = tf.scatter_nd_update(memory, idx, mem_t)

        # memory[:, :] = mem_t

        # update states
        mem_t = K.reshape(mem_t, (1, self.units))
        new_states = states[:((self.memory_size[0] * self.memory_size[1]) + 1)]
        new_states[0] = r_t
        new_states[((x-1) * self.units) + y] = mem_t

        return c_t, new_states

class NeuralMap(RNN):
    """Neural Map Implementation4.
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

    @interfaces.legacy_recurrent_support
    def __init__(self, units, pos_input,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 memory_size=[10,10],
                 **kwargs):
        cell = NeuralMapCell(units, pos_input=pos_input,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        memory_size=memory_size)
        super(NeuralMap, self).__init__(cell, return_sequences=True, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(NeuralMap, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def memory_size(self):
        return self.cell.memory_size

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'memory_size': self.memory_size}
        base_config = super(NeuralMap, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))