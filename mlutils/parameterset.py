# -*- coding: utf-8 -*-
# mostly taken from Breze

import os
import sys
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda

try:
    gpu_environ = os.environ['BREZE_PARAMETERSET_DEVICE']
    if gpu_environ == 'gpu':
        GPU = True
    elif gpu_environ == 'cpu':
        GPU = False
    else:
        print "BREZE_PARAMETERSET_DEVICE must be either 'cpu' or 'gpu'"
        sys.exit(1)
except KeyError:
    GPU = theano.config.device == 'gpu'

if GPU:
    import gnumpy



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
            region = self._data[n_used:n_used + size]
            # Then shape it correctly and make it accessible from the outside.
            region = region.reshape(shape)
            self.views[key] = region

            # Get the right variable as a subtensor.
            var = self._flat[n_used:n_used + size].reshape(shape)
            var.name = key
            setattr(self, key, var)

            n_used += size

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


