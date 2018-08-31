# -*- coding: utf-8 -*-
"""This tensorflow extension implements a CTRNN in order to build MTRNNs."""

import collections
import math
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl

print("Warning: model implemented for python 3.5 + tensorflow 1.7")

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_CTRNNStateTuple = collections.namedtuple("CTRNNStateTuple", ("z", "y"))


class CTRNNStateTuple(_CTRNNStateTuple):
    """
    Tuple used by CTRNN Cells for `state_size`,
    `zero_state` (in outher wrappers), and as an output state.

    Stores two elements: `(z, y)`, in that order.
    Where `z` is the internal state and `y` is the output.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (z, y) = self
        if z.dtype != y.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                                            (str(z.dtype), str(y.dtype)))
        return z.dtype


class CTRNNCell(rnn_cell_impl.LayerRNNCell):
    """Continuous Time RNN cell.

    Args:
        num_units: int, the number of units in the RNN cell;
            or array, the number of units per module in the RNN cell.
        tau: timescale (or unit-dependent time constant of leakage).
        state_is_tuple: cell maintains two state - an internal state (z).
        use_bias: enable or disable biases.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.    If not `True`, and the existing scope already has
            the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self, num_units, tau=1., state_is_tuple=True, initializer=None,
                 use_bias=True, activation=None, reuse=None, name=None):
        super(CTRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = rnn_cell_impl.base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._tau = array_ops.constant(
            tau, dtype=self.dtype, shape=[self._num_units],
            name="taus")
        self._state_is_tuple = state_is_tuple
        self._initializer = initializer
        self._use_bias = use_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (CTRNNStateTuple(self._num_units, self._num_units)
                        if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._initializer)

        if self._use_bias:
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):

        if self._state_is_tuple:
            prev_z, prev_y = state
        else:
            prev_z, prev_y = array_ops.split(
                value=state, num_or_size_splits=2,
                axis=constant_op.constant(1, dtype=dtypes.int32))

        x = math_ops.matmul(array_ops.concat([inputs, prev_y], 1), self._kernel)

        if self._use_bias:
            x = nn_ops.bias_add(x, self._bias)

        z = (1. - 1. / self._tau) * prev_z + (1. / self._tau) * x

        y = self._activation(z)

        if self._state_is_tuple:
            new_state = CTRNNStateTuple(z, y)
        else:
            new_state = array_ops.concat([z, y], 1)
        return y, new_state


class MultipleCTRNNCell(rnn_cell_impl.LayerRNNCell):
    """Continuous Time RNN cell.

    Args:
        num_units_v: int, the number of units in the RNN cell;
            or array, the number of units per module in the RNN cell.
        tau_v: timescale (or unit-dependent time constant of leakage).
        num_modules: int, the number of modules - only used when num_units_v not a vector
        connectivity: connection scheme in case of more than one modules
        state_is_tuple: cell maintains two state - an internal state (z).
        use_bias: enable or disable biases.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.    If not `True`, and the existing scope already has
            the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self, num_units_v, tau_v=1., num_modules=None,
                 connectivity='dense', state_is_tuple=True, initializer=None,
                 use_bias=True, activation=None, reuse=None, name=None):
        super(MultipleCTRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = rnn_cell_impl.base_layer.InputSpec(ndim=2)
        self._connectivity = connectivity
        if isinstance(num_units_v, list):
            self._num_units_v = num_units_v[:]
            self._num_modules = len(num_units_v)
            self._num_units = 0
            for k in range(self._num_modules):
                self._num_units += num_units_v[k]
        else:
            self._num_units = num_units_v
            if num_modules > 1:
                self._num_modules = int(num_modules)
                self._num_units_v = [num_units_v//num_modules for k in range(num_modules)]
            else:
                self._num_modules = 1
                self._num_units_v = [num_units_v]
                self._connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_v, list):
            if len(tau_v) != self._num_modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            self._tau = array_ops.constant(
                [[max(1., tau_v[k])] for k in range(self._num_modules) for n in range(self._num_units_v[k])],
                dtype=self.dtype, shape=[self._num_units],
                name="taus")
        else:
            self._tau = array_ops.constant(
                max(1., tau_v), dtype=self.dtype, shape=[self._num_units],
                name="taus")

        self._state_is_tuple = state_is_tuple
        self._initializer = initializer
        self._use_bias = use_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (CTRNNStateTuple(self._num_units, self._num_units)
                        if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value

        if self._connectivity == 'partitioned':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[input_depth + self._num_units_v[k],
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        elif self._connectivity == 'clocked':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[input_depth + sum(self._num_units_v[k:self._num_modules]),
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        elif self._connectivity == 'adjacent':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[input_depth + sum(self._num_units_v[max(0, k - 1):min(self._num_modules, k + 1 + 1)]),
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        else:  # == 'dense'
            self._kernel_v = [self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._initializer)]

        if self._use_bias:
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):

        if self._state_is_tuple:
            prev_z, prev_y = state
        else:
            prev_z, prev_y = array_ops.split(
                value=state, num_or_size_splits=2,
                axis=constant_op.constant(1, dtype=dtypes.int32))
        prev_y_v = array_ops.split(prev_y, self._num_units_v, axis=1)

        if self._connectivity == 'partitioned':
            x = array_ops.concat([math_ops.matmul(
                    array_ops.concat([inputs, prev_y_v[k]], 1),
                    self._kernel_v[k]) for k in range(self._num_modules)], 1)
        elif self._connectivity == 'clocked':
            x = array_ops.concat([math_ops.matmul(
                     array_ops.concat([inputs, array_ops.concat(prev_y_v[k:self._num_modules], 1)], 1),
                     self._kernel_v[k]) for k in range(self._num_modules)], 1)
        elif self._connectivity == 'adjacent':
            x = array_ops.concat([math_ops.matmul(
                    array_ops.concat([inputs, array_ops.concat(prev_y_v[max(0, k - 1):min(self._num_modules, k + 1 + 1)], 1)], 1),
                    self._kernel_v[k]) for k in range(self._num_modules)], 1)
        else:  # 'dense'
            x = math_ops.matmul(array_ops.concat([inputs, prev_y], 1), self._kernel_v[0])

        if self._use_bias:
            x = nn_ops.bias_add(x, self._bias)

        z = (1. - 1. / self._tau) * prev_z + (1. / self._tau) * x

        y = self._activation(z)

        if self._state_is_tuple:
            new_state = CTRNNStateTuple(z, y)
        else:
            new_state = array_ops.concat([z, y], 1)
        return y, new_state


class MultipleTimescaleRNNCell(rnn_cell_impl.LayerRNNCell):
    """Multiple Timescale RNN cell.

    Args:
        num_units_v: int, the number of units in the RNN cell;
            or array, the number of units per module in the RNN cell.
        tau_v: timescale (or unit-dependent time constant of leakage).
        num_modules: int, the number of modules - only used when num_units_v not a vector
        connectivity: connection scheme in case of more than one modules
        state_is_tuple: cell maintains two state - an internal state (z).
        use_bias: enable or disable biases.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.    If not `True`, and the existing scope already has
            the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self, num_units_v, tau_v=1., num_modules=None,
                 connectivity='dense', state_is_tuple=True, initializer=None,
                 use_bias=True, activation=None, reuse=None, name=None):
        super(MultipleTimescaleRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = rnn_cell_impl.base_layer.InputSpec(ndim=2)
        self._connectivity = connectivity
        if isinstance(num_units_v, list):
            self._num_units_v = num_units_v[:]
            self._num_modules = len(num_units_v)
            self._num_units = 0
            for k in range(self._num_modules):
                self._num_units += num_units_v[k]
        else:
            self._num_units = num_units_v
            if num_modules > 1:
                self._num_modules = int(num_modules)
                self._num_units_v = [num_units_v//num_modules for k in range(num_modules)]
            else:
                self._num_modules = 1
                self._num_units_v = [num_units_v]
                self._connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_v, list):
            if len(tau_v) != self._num_modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            self._tau = array_ops.constant(
                [[max(1., tau_v[k])] for k in range(self._num_modules) for n in range(self._num_units_v[k])],
                dtype=self.dtype, shape=[self._num_units],
                name="taus")
        else:
            self._tau = array_ops.constant(
                max(1., tau_v), dtype=self.dtype, shape=[self._num_units],
                name="taus")

        self._state_is_tuple = state_is_tuple
        self._initializer = initializer
        self._use_bias = use_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (CTRNNStateTuple(self._num_units, self._num_units)
                        if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value
        #input_depth = inputs_shape[1][0].value

        if self._connectivity == 'partitioned':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[(input_depth if k is 0 else 0) + self._num_units_v[k],
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        elif self._connectivity == 'clocked':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[(input_depth if k is 0 else 0) + sum(self._num_units_v[k:self._num_modules]),
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        elif self._connectivity == 'adjacent':
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[(input_depth if k is 0 else 0) + sum(self._num_units_v[max(0, k - 1):min(self._num_modules, k + 1 + 1)]),
                           self._num_units_v[k]],
                    initializer=self._initializer)]
        else:  # == 'dense'
            # self._kernel_v = [self.add_variable(
            #     _WEIGHTS_VARIABLE_NAME,
            #     shape=[input_depth + self._num_units, self._num_units],
            #     initializer=self._initializer)]
            self._kernel_v = []
            for k in range(self._num_modules):
                self._kernel_v += [self.add_variable(
                    _WEIGHTS_VARIABLE_NAME + str(k),
                    shape=[(input_depth if k is 0 else 0) + sum(self._num_units_v[:]),
                           self._num_units_v[k]],
                    initializer=self._initializer)]

        if self._use_bias:
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
    #def call(self, inputscsc, state):

        # inputsec includes input to the first module only as well as
        #(inputs, csc) = inputscsc

        if self._state_is_tuple:
            prev_z, prev_y = state
        else:
            prev_z, prev_y = array_ops.split(
                value=state, num_or_size_splits=2,
                axis=constant_op.constant(1, dtype=dtypes.int32))
        prev_y_v = array_ops.split(prev_y, self._num_units_v, axis=1)

        if self._connectivity == 'partitioned':
            x = array_ops.concat([math_ops.matmul(
                    array_ops.concat([inputs, prev_y_v[k]], 1),
                    self._kernel_v[k]) if k is 0 else math_ops.matmul(
                    array_ops.concat([prev_y_v[k]], 1),
                    self._kernel_v[k]) for k in range(self._num_modules)], 1)
        elif self._connectivity == 'clocked':
            x = array_ops.concat([math_ops.matmul(
                     array_ops.concat([inputs, array_ops.concat(prev_y_v[k:self._num_modules], 1)], 1),
                     self._kernel_v[k]) if k is 0 else math_ops.matmul(
                     array_ops.concat([array_ops.concat(prev_y_v[k:self._num_modules], 1)], 1),
                     self._kernel_v[k]) for k in range(self._num_modules)], 1)
        elif self._connectivity == 'adjacent':
            x = array_ops.concat([math_ops.matmul(
                    array_ops.concat([inputs, array_ops.concat(prev_y_v[max(0, k - 1):min(self._num_modules, k + 1 + 1)], 1)], 1),
                    self._kernel_v[k]) if k is 0 else math_ops.matmul(
                    array_ops.concat([array_ops.concat(prev_y_v[max(0, k - 1):min(self._num_modules, k + 1 + 1)], 1)], 1),
                    self._kernel_v[k]) for k in range(self._num_modules)], 1)
        else:  # 'dense'
            #x = math_ops.matmul(array_ops.concat([inputs, prev_y], 1), self._kernel_v[0])
            x = array_ops.concat([math_ops.matmul(
                     array_ops.concat([inputs, array_ops.concat(prev_y_v[:], 1)], 1),
                     self._kernel_v[k]) if k is 0 else math_ops.matmul(
                     array_ops.concat([array_ops.concat(prev_y_v[:], 1)], 1),
                     self._kernel_v[k]) for k in range(self._num_modules)], 1)

        if self._use_bias:
            x = nn_ops.bias_add(x, self._bias)

        z = (1. - 1. / self._tau) * prev_z + (1. / self._tau) * x

        y = self._activation(z)

        if self._state_is_tuple:
            new_state = CTRNNStateTuple(z, y)
        else:
            new_state = array_ops.concat([z, y], 1)

        #return only the output of the first module (the input-output layer)
        return y, new_state
        #return array_ops.slice(y, [0, 0], [-1, self._num_units_v[0]]), new_state
        #return array_ops.slice(y, [0, 0], [-1, inputs.shape[1]]), new_state

# THE FOLLOWING WERE MADE FOR TENSORFLOW 1.4.0 AND NEED TO GET REVISED

# class MTRNNLayers():
#     """
#     MTRNN layers according to Yamashita & Tani 2008
#     build by arbitrary many horizonal CTRNN cells
#
#     Args:
#         num_units: tensor, The number of units of all layers.
#         tau: timescale of all layers
#         csc_units: int, The number of context controlling units
#         and an output (y) - either in a tuple or an concatenated array
#         activation: Nonlinearity to use.    Default: `tanh`.
#         reuse: (optional) Python boolean describing whether to reuse
#         variables in an existing scope.
#         If not `True`, and the existing scope already has the given
#         variables, an error is raised.
#     """
#
#     def __init__(self, nums_units, taus=1., state_is_tuple= True,
#                  activation=None, reuse=None):
#         super(CTRNNCell, self).__init__(_reuse=reuse)
#         self._numss_units = nums_units
#         self._taus = taus
#         self._state_is_tuple = state_is_tuple
#         self._activation = activation or math_ops.tanh
#
#     def __call__(self, inputs, state, scope=None):