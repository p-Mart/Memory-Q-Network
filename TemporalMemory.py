# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import functools
import warnings

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
from keras.layers.recurrent import RNN

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces

import tensorflow as tf


class SimpleMemoryCell(Layer):
    """Cell class for the LSTM layer.
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
            Set equal to the timesteps.
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 memory_size=10,
                 **kwargs):
        super(SimpleMemoryCell, self).__init__(**kwargs)
        self.units = units

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)

        self.memory_size = memory_size
        self.state_size = (self.units, ) * \
            (self.memory_size * 2)  # M_key, M_value

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_key = self.add_weight(
            shape=(input_dim - self.units, self.units),
            name='W_key',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.W_value = self.add_weight(
            shape=(input_dim - self.units, self.units),
            name='W_value',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # super(SimpleMemoryCell, self).build(input_shape)
        self.built = True

    def call(self, inputs, states, training=None):
        # inputs is e_t concatenated with h_t

        e_t = inputs[:, :-self.units]
        h_t = inputs[:, -self.units:]

        # states is M_key, M_value
        M_key = list(states[:self.memory_size])  # shape=[Mxm]
        M_value = list(states[self.memory_size:])  # shape=[Mxm]

        # Conversion to tensors
        M_key_tens = K.stack(M_key, axis=1)
        M_value_tens = K.stack(M_value, axis=1)

        # Calculate attention for memory
        at = K.softmax(K.batch_dot(M_key_tens, h_t, axes=[2, 1]))

        # Read from memory using attention, give as output
        output = K.batch_dot(M_value_tens, at, axes=[1, 1])

        # Update states
        M_key.pop(0)  # shape = [M-1xm]
        M_value.pop(0)  # shape = [M-1xm]
        m_key = K.dot(e_t, self.W_key)  # shape = [1xm]
        m_value = K.dot(e_t, self.W_value)  # shape = [1xm]
        M_key.append(m_key)  # shape = [Mxm]
        M_value.append(m_value)  # shape = [Mxm]

        return output, M_key + M_value


'''
    def get_config(self):
        config={'memory_size' : self.memory_size}
        base_config = super(SimpleMemoryCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
'''


class SimpleMemory(RNN):
    """Long-Short Term Memory layer - Hochreiter 1997.
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
            Set equal to the timesteps.
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 memory_size=10,
                 **kwargs):

        cell = SimpleMemoryCell(units,
                                kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer,
                                kernel_regularizer=kernel_regularizer,
                                recurrent_regularizer=recurrent_regularizer,
                                kernel_constraint=kernel_constraint,
                                recurrent_constraint=recurrent_constraint,
                                memory_size=memory_size)
        super(SimpleMemory, self).__init__(cell=cell, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(SimpleMemory, self).call(inputs,
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
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'memory_size': self.memory_size}

        base_config = super(SimpleMemory, self).get_config()

        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
