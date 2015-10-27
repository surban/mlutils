# -*- coding: utf-8 -*-
# mostly taken from Breze

import os
import sys
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
from . import gpu
from mlutils import xp
from mlutils.gpu import post

GPU = gpu.GPU
if GPU:
    import gnumpy
    from gnumpy import garray


class ParameterSet(object):
    """ParameterSet class.

    This class provides functionality to group several Theano tensors of
    different sizes in a consecutive chunk of memory. The main aim of this is
    to allow a view on several tensors as a single long vector.

    In the following, a (parameter) array refers to a concrete instantiation of
    a parameter variable (with concrete values) while a (parameter)
    tensor/variable refers to the symbolic Theano variable.


    Parameters
    ----------

    Initialization takes a variable amount of keyword arguments, where each has
    to be a single integer or a tuple of arbitrary length containing only
    integers. For each of the keyword argument keys a tensor of the shape given
    by the value will be created. The key is the identifier of that variable.


    Attributes
    ----------

    n_pars : integer
        Total amount of parameters.

    flat : Theano vector
        Flat one dimensional tensor containing all the different tensors
        flattened out. Symbolic pendant to ``data``.

    data : array_like
        Concrete array containig all the different arrays flattened out.
        Concrete pendant to ``flat``.

    views : dictionary
        All parameter arrays can be accessed by with their identifier as key
        in this dictionary.

    vars : dictionary
        All Theano vectors can be accessed by with their identifier as key
        in this dictionary.

    All symbolic variables can be accessed as attributes of the object, all
    concrete variables as keys. E.g. parameter_set.x references the symbolic
    variable, while parameter_set['x'] will give you the concrete array.
    """

    def __init__(self, **kwargs):
        # Make sure all size specifications are tuples.
        kwargs = dict((k, v if isinstance(v, tuple) else (v,))
                      for k, v in kwargs.iteritems())

        # Find out total size of needed parameters and create memory for it.
        sizes = [np.prod(i) for i in kwargs.values()]
        self.n_pars = sum(sizes)

        # print statistics
        print "Model parameter sizes: "
        print "Number of model parameters: %d" % self.n_pars

        # Create two representations of the parameters of the object. The first
        # is the symbolic theano variable (of which the type is GPU/CPU
        # specific), the second either a gnumpy or numpy array (depending on
        # GPU/CPU again). Also set a default size for testing.
        if GPU:
            self._data = gnumpy.zeros(self.n_pars)
            self._flat = theano.sandbox.cuda.fvector('parameters')
        else:
            self._data = np.empty(self.n_pars).astype(theano.config.floatX)
            self._flat = T.vector('parameters')

        self._flat.tag.test_value = self._data

        # Go through parameters and assign space and variable.
        self.views = {}
        n_used = 0 	# Number of used parameters.

        for (key, shape), size in zip(kwargs.items(), sizes):
            # Make sure the key is legit -- that it does not overwrite
            # anything.
            if hasattr(self, key):
                raise ValueError("%s is an illegal name for a variable")

            # Get the region from the big flat array.
            region = self._data[n_used : n_used + size]
            # Then shape it correctly and make it accessible from the outside.
            region = region.reshape(shape)
            self.views[key] = region

            # Get the right variable as a subtensor.
            var = self._flat[n_used:n_used + size].reshape(shape)
            var.name = key
            setattr(self, key, var)

            n_used += size

        self.constants = {}
        """Constant values for variables.
        restore_constants must be called after every update to ensure that constants have
        their requested value."""

    @property
    def vars(self):
        """All Theano vectors can be accessed by with their identifier as key
        in this dictionary."""
        return {k: getattr(self, k) for k in self.views.iterkeys()}

    @property
    def data(self):
        """Actual numerical data values (numpy array) of this ParameterSet"""
        return self._data

    @property
    def flat(self):
        """Symbolic Theano variable corresponding to the values of this ParameterSet"""
        return self._flat

    def __contains__(self, key):
        return key in self.views

    def __getitem__(self, key):
        return self.views[key]

    def __setitem__(self, key, value):
        self.views[key][:] = value

    def var_at_index(self, index):
        """
        Returns the variable name, the value at the given index belongs to, the index within
        that variable is appended to the variable name.
        :param index: index within flat view
        :return: "variable_name[index]"
        """
        index = int(index)
        if not (0 <= index < self._data.size):
            raise ValueError("index %d out of bounds" % index)
        iter_index = 0
        for param in self.views:
            if index < iter_index + self.views[param].size:
                return '%s[%i]' % (param, index - iter_index)
            iter_index += self.views[param].size

    def indices_at_var(self, key):
        """Returns a tuple containing the index range of the given variable name.
        :param key: variable name
        :return: (start_index, end_index+1)
        """
        if key not in self.views:
            raise ValueError("ParameterSet does not contain %s" % key)
        iter_index = 0
        for param in self.views:
            if param == key:
                return (iter_index, iter_index + self.views[param].size)
            iter_index += self.views[param].size

    def find_large_gradient_vars(self, grad, threshold=1.0):
        """
        Finds elements within the gradient that exceed a given threshold.
        :param grad: the gradient w.r.t. to this ParameterSet
        :param threshold: threshold for large value
        :return: a dict of large value gradient elements (resolved to variables names)
                 and their respective values
        """
        assert grad.size == self.data.size, 'grad should be calculated with respect to variables in ps'
        params = {}
        for i in range(grad.size):
            if abs(grad[i]) >= threshold:
                params.update({i: (self.var_at_index(i), grad[i])})
        return params

    def split_gradient(self, grad):
        """
        Splits the passed gradient into individual gradients w.r.t. to the variables of this ParameterSet.
        :param grad: the gradient w.r.t. to this ParameterSet
        :return: a dict of (variable name, gradient) pairs
        """
        assert grad.size == self.data.size, 'grad should be calculated with respect to variables in ps'

        var_grad = {}
        pos = 0
        for param in self.views:
            size = self.views[param].size
            var_grad[param] = grad[pos : pos + size]
            pos += size
        return var_grad

    def restore_constants(self):
        """
        Ensures that all constant variables have their required values.
        """
        for var, value in self.constants.iteritems():
            if isinstance(value, (int, float)):
                value = value * xp.ones(self[var].shape)
            if GPU and not isinstance(value, garray):
                raise TypeError("constant value for variable %s is not a garray although this "
                                "ParameterSet is stored on the GPU" % var)
            self[var] = value

    def nullify_gradient_of_constants(self, grad):
        """
        Sets the elements corresponding to constants in this ParameterSet to zero in the gradient inplace.
        :param grad: the gradient
        :return: the gradient with zero elements for constants
        """
        if len(self.constants) == 0:
            return grad
        rngs = [self.indices_at_var(var) for var in self.constants.iterkeys()]
        idxs = [np.arange(*rng) for rng in rngs]
        sel = reduce(lambda a, b: np.concatenate((a,b)), idxs, [])
        # print "setting gradient elements to zero: ", sel
        sel = post(sel)
        if GPU:
            grad[sel] = gnumpy.zeros((len(sel),))
        else:
            grad[sel] = 0
        return grad


