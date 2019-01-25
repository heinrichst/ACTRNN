# -*- coding: utf-8 -*-
"""
This tensorflow extension implements ACTRNN, VCTRNN, AVCTRNN.
Heinrich et al. 2018
Updated Jan 2019 for tf_1.12
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.recurrent \
    import _generate_dropout_mask, _generate_zero_filled_state_for_cell, RNN
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


_MAX_TIMESCALE = 999999
_MAX_SIGMA = 500000
_ALMOST_ONE = 0.999999
_ALMOST_ZERO = 0.000001


@tf_export('keras.layers.ACTRNNCell')
class ACTRNNCell(Layer):
    """Cell class for ACTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ACTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., tau_vec[k])] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [tau_vec for k in range(self.modules)]
            else:
                self.tau_vec = [tau_vec]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_tau_initializer = initializers.get(w_tau_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_tau_regularizer = regularizers.get(w_tau_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_tau_constraint = constraints.get(w_tau_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.w_tau = self.add_weight(
            shape=(self.units,),
            name='wtimescales',
            initializer=self.w_tau_initializer,
            regularizer=self.w_tau_regularizer,
            constraint=self.w_tau_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:  # 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])

        taus_act = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        return y, [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(ACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.ACTRNN')
class ACTRNN(RNN):
    """Adaptive Continuous Time RNN that can have multiple modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `ACTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = ACTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(ACTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(ACTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_tau_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(ACTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.VCTRNNCell')
class VCTRNNCell(Layer):
    """Cell class for VCTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        max_sigma_vec: Positive float >= 0 <= tau, maximal deviation of tau.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 max_sigma_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(VCTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., tau_vec[k])] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [tau_vec for k in range(self.modules)]
            else:
                self.tau_vec = [tau_vec]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        if isinstance(max_sigma_vec, list):
            if len(max_sigma_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if max_sigma_vec[k] < 0 or max_sigma_vec[k] > self.tau_vec[k]:
                    raise ValueError("timescale sigmas must be equal or "
                                     "larger 0 and equal or smaller tau")
            self.max_sigma_vec = max_sigma_vec[:]
            self.sigmas = array_ops.constant(
                [[max(0., max_sigma_vec[k])*n/max(1., self.units_vec[k] - 1)]
                 for k in range(self.modules) for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="sigmas")
        else:
            if max_sigma_vec < 0 or max_sigma_vec > self.tau_vec[0]:
                raise ValueError("timescale sigmas must be equal or "
                                 "larger 0 and equal or smaller tau")
            if self.modules > 1:
                self.max_sigma_vec = [max_sigma_vec
                                      for k in range(self.modules)]
            else:
                self.max_sigma_vec = [max_sigma_vec]
            self.sigmas = array_ops.constant(
                max(0., max_sigma_vec), dtype=self.dtype, shape=[self.units],
                name="sigmas")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:  # 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])


        epsilon = K.random_normal(array_ops.shape(self.sigmas), 0,
                                  1, dtype=self.dtype)

        taus_act = K.clip_ops.clip_by_value(self.taus + self.sigmas * epsilon,
                                            1., _MAX_TIMESCALE)

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        return y, [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return self.taus

    def get_sigmas(self):
        return self.sigma

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'max_sigma_vec':
                self.max_sigma_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(VCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.VCTRNN')
class VCTRNN(RNN):
    """Variational Continuous Time RNN that can have multiple modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        max_sigma_vec: Positive float >= 0 <= tau, maximal deviation of tau.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 max_sigma_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `VCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = VCTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            max_sigma_vec=max_sigma_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(VCTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(VCTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def max_sigma_vec(self):
        return self.cell.max_sigma_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'max_sigma_vec':
                self.max_sigma_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(VCTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.AVCTRNNCell')
class AVCTRNNCell(Layer):
    """Cell class for AVCTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        max_sigma_vec: Positive float >= 0 <= tau, maximal deviation of tau.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        w_sigma_initializer: Initializer for the w_sigma vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        w_sigma_regularizer: Regularizer function applied to the w_sigma vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        w_sigma_constraint: Constraint function applied to the w_sigma vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 max_sigma_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 w_sigma_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 w_sigma_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 w_sigma_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(AVCTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., tau_vec[k])] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [tau_vec for k in range(self.modules)]
            else:
                self.tau_vec = [tau_vec]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        if isinstance(max_sigma_vec, list):
            if len(max_sigma_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if max_sigma_vec[k] < 0 or max_sigma_vec[k] > self.tau_vec[k]:
                    raise ValueError("timescale sigmas must be equal or "
                                     "larger 0 and equal or smaller tau")
            self.max_sigma_vec = max_sigma_vec[:]
            self.sigmas = array_ops.constant(
                [[max(0., max_sigma_vec[k])*n/max(1., self.units_vec[k] - 1)]
                 for k in range(self.modules) for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="sigmas")
        else:
            if max_sigma_vec < 0 or max_sigma_vec > self.tau_vec[0]:
                raise ValueError("timescale sigmas must be equal or "
                                 "larger 0 and equal or smaller tau")
            if self.modules > 1:
                self.max_sigma_vec = [max_sigma_vec
                                      for k in range(self.modules)]
            else:
                self.max_sigma_vec = [max_sigma_vec]
            self.sigmas = array_ops.constant(
                max(0., max_sigma_vec), dtype=self.dtype, shape=[self.units],
                name="sigmas")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_tau_initializer = initializers.get(w_tau_initializer)
        self.w_sigma_initializer = initializers.get(w_sigma_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_tau_regularizer = regularizers.get(w_tau_regularizer)
        self.w_sigma_regularizer = regularizers.get(w_sigma_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_tau_constraint = constraints.get(w_tau_constraint)
        self.w_sigma_constraint = constraints.get(w_sigma_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.w_tau = self.add_weight(
            shape=(self.units,),
            name='wtimescales',
            initializer=self.w_tau_initializer,
            regularizer=self.w_tau_regularizer,
            constraint=self.w_tau_constraint)

        self.w_sigma = self.add_weight(
            shape=(self.units,),
            name='wsigmas',
            initializer=self.w_sigma_initializer,
            regularizer=self.w_sigma_regularizer,
            constraint=self.w_sigma_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        self.log_sigmas = K.log(self.sigmas + _ALMOST_ZERO)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:  # 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])

        sigmas_act = K.clip_ops.clip_by_value(
            K.exp(self.w_sigma + self.log_sigmas) - _ALMOST_ZERO,
            0., _MAX_SIGMA)

        epsilon = K.random_normal(array_ops.shape(sigmas_act), 0,
                                  1, dtype=self.dtype)

        taus_act = K.clip_ops.clip_by_value(
            K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE
            + sigmas_act * epsilon,
            1., _MAX_TIMESCALE)

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        return y, [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE \
            if self.built else None

    def get_sigmas(self):
        return K.exp(self.w_sigma + self.log_sigmas) - _ALMOST_ZERO \
            if self.built else None

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'w_sigma_initializer':
                initializers.serialize(self.w_sigma_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'w_sigma_regularizer':
                regularizers.serialize(self.w_sigma_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'w_sigma_constraint':
                constraints.serialize(self.w_sigma_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AVCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.AVCTRNN')
class AVCTRNN(RNN):
    """Adaptive Variational Continuous Time RNN that can have multiple modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        max_sigma_vec: Positive float >= 0 <= tau, maximal deviation of tau.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        w_sigma_initializer: Initializer for the w_sigma vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        w_sigma_regularizer: Regularizer function applied to the w_sigma vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        w_sigma_constraint: Constraint function applied to the w_sigma vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 max_sigma_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 w_sigma_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 w_sigma_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 w_sigma_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `AVCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = AVCTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            max_sigma_vec=max_sigma_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            w_sigma_initializer=w_sigma_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            w_sigma_regularizer=w_sigma_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            w_sigma_constraint=w_sigma_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(AVCTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(AVCTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def max_sigma_vec(self):
        return self.cell.max_sigma_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def w_sigma_initializer(self):
        return self.cell.w_sigma_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def w_sigma_regularizer(self):
        return self.cell.w_sigma_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_tau_constraint

    @property
    def w_sigma_constraint(self):
        return self.cell.w_sigma_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'max_sigma_vec':
                self.max_sigma_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'w_sigma_initializer':
                initializers.serialize(self.w_sigma_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'w_sigma_regularizer':
                regularizers.serialize(self.w_sigma_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'w_sigma_constraint':
                constraints.serialize(self.w_sigma_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AVCTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
