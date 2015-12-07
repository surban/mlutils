# -*- coding: utf-8 -*-
# taken from Breze, but now rewritten

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
from mlutils import xp
from mlutils.gpu import post, GPU
from operator import add

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

    _num_var_slices : dictionary
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
        self._constants = {}
        self._constants_selector = []
        self._constants_zeros = []

        # extract partition specification
        if 'partitions' in kwargs:
            partitions = dict(kwargs['partitions'])
            del kwargs['partitions']
        else:
            partitions = {}

        # extract variable shapes as tuples and verify
        shapes = dict((k, v if isinstance(v, tuple) else (v,)) for k, v in kwargs.iteritems())
        for name in shapes.iterkeys():
            if not isinstance(name, basestring):
                raise TypeError("variable name must be a string")
            if hasattr(self, name):
                raise ValueError("variable name %s is illegal because it would override an object attribute" % name)

        # verify partitions
        for name, members in partitions.iteritems():
            if not isinstance(name, basestring):
                raise TypeError("partition name must be a string")
            if not isinstance(members, list):
                raise TypeError("partition member specification must be a list of variable names")
            for var in members:
                if not isinstance(var, basestring):
                    raise TypeError("partition member specification must be a list of variable names")

        # put vars without partition specification into _default partition
        vars_in_partitions = reduce(add, partitions.itervalues(), [])
        for var in vars_in_partitions:
            if var not in shapes:
                raise ValueError("variable %s from partition specification has no shape specification" % var)
            if vars_in_partitions.count(var) != 1:
                raise ValueError("variable %s appears in multiple parameter partitions" % var)
        all_vars = shapes.keys()
        remaining_vars = list(set(all_vars) - set(vars_in_partitions))
        if len(remaining_vars) > 0:
            partitions['_default'] = remaining_vars

        # sort partitions and their members to obtain a stable ordering
        partition_order = sorted(partitions.keys())
        for partition, members in partitions.iteritems():
            partitions[partition] = sorted(members)

        # compute layout of variables and partitions
        pos = 0
        self._var_layout = {}
        self._part_layout = {}
        for partition in partition_order:
            part_start = pos
            for var in partitions[partition]:
                size = np.prod(shapes[var])
                self._var_layout[var] = (pos, pos + size)
                pos += size
            self._part_layout[partition] = (part_start, pos)
        self.n_pars = pos

        # Find out total size of needed parameters and create memory for it.
        sizes = [np.prod(i) for i in shapes.values()]

        # Create two representations of the parameters of the object. The first
        # is the symbolic Theano variable, the second is a numeric array.
        if GPU:
            self._num_data = gnumpy.zeros(self.n_pars)
            # we actually need to create a standard Theano variable (not theano.sandbox.cuda.fvector)
            # because translation to GPU variables will be performed by gpu.function before compilation
            self._sym_data = T.vector('parameters')
        else:
            self._num_data = np.empty(self.n_pars).astype(theano.config.floatX)
            self._sym_data = T.vector('parameters')
        self._sym_data.tag.test_value = self._num_data

        # assign numeric and symbolic slices to variables
        self._num_var_slices = {}
        self._sym_var_slices = {}
        for var, shape in shapes.iteritems():
            start, stop = self._var_layout[var]
            self._num_var_slices[var] = self._num_data[start : stop].reshape(shape)
            self._sym_var_slices[var] = self._sym_data[start : stop].reshape(shape)
            self._sym_var_slices[var].name = var

        # assign numeric and symbolic slices to partitions
        self._num_part_slices = {}
        self._sym_part_slices = {}
        for part, (start, stop) in self._part_layout.iteritems():
            self._num_part_slices[part] = self._num_data[start : stop]
            self._sym_part_slices[part] = self._sym_data[start : stop]
            self._sym_part_slices[part].name = 'partition_' + part

        # print statistics
        print "Model parameters: "
        print "Total count:      %d" % self.n_pars
        print "Partitions:       %s" % repr(partitions)

    @property
    def partitions(self):
        """Partitions in this ParameterSet."""
        return self._part_layout.keys()

    ###########################################################################
    # numeric/symbolic access to variables
    ###########################################################################

    def sym(self, var):
        """
        Symbolic variable.
        :param var: variable name
        :return: symbolic variable
        """
        return self._sym_var_slices[var]

    def num(self, var):
        """
        Numeric variable.
        :param var: variable name
        :return: numeric variable view (alias)
        """
        return self._num_var_slices[var]

    @property
    def sym_vars(self):
        """Dicitionary of variable names and their corresponding symbolic variables."""
        return self._sym_var_slices.copy()

    @property
    def num_vars(self):
        """Dicitionary of variable names and their corresponding numeric variables."""
        return self._num_var_slices.copy()

    @property
    def sym_data(self):
        """Symbolic variable corresponding to all values of this ParameterSet in a flat array."""
        return self._sym_data

    @property
    def num_data(self):
        """All numerical data values of this ParameterSet in a flat array."""
        return self._num_data

    def num_partition(self, partiton):
        """
        Flat numeric vector of all variable values in the specified partition.
        :param partiton: partition
        :return: flat array of numeric values
        """
        return self._num_part_slices[partiton]

    ###########################################################################
    # variable localization within flat vector
    ###########################################################################

    def var_at(self, index):
        """
        Returns the variable name the value at the given index belongs to and the corresponding
        index within that variable.
        :param index: index within self.num_data
        :return: (variable_name, index_within)
        """
        index = int(index)
        if not (0 <= index < self._num_data.size):
            raise ValueError("index %d out of bounds" % index)
        for var, (start, stop) in self._var_layout.iteritems():
            if start <= index < stop:
                return var, index - start

    def extents_of_var(self, var):
        """
        Returns a tuple containing the index range of the given variable name.
        :param var: variable name
        :return: (start_index, end_index + 1)
        """
        return self._var_layout[var]

    def extents_of_partition(self, part):
        """
        Returns a tuple containing the index range of the given partition name.
        :param part: partition name
        :return: (start_index, end_index + 1)
        """
        return self._part_layout[part]

    ###########################################################################
    # flat vector processing
    ###########################################################################

    def find_large_elements(self, data, threshold=1.0):
        """
        Finds elements within a flat numeric data array that exceed a given threshold.
        :param data: a data array of same shape as self.num_data (e.g. the gradient w.r.t. to this ParameterSet)
        :param threshold: threshold for large value
        :return: a dict of large value elements (resolved to variables names) and their respective values
        """
        if not data.shape == self.num_data.shape:
            raise ValueError("shape of specified data does not match shape of this ParameterSet's data")
        return {idx: (self.var_at(idx), data[idx])
                for idx in range(data.size) if abs(data[idx]) >= threshold}

    def split(self, data):
        """
        Splits the passed data into individual elements w.r.t. to the variables of this ParameterSet.
        :param data: a data array of same shape as self.num_data (e.g. the gradient w.r.t. to this ParameterSet)
        :return: a dict of (variable name, elements) pairs
        """
        if not data.shape == self.num_data.shape:
            raise ValueError("shape of specified data does not match shape of this ParameterSet's data")
        vars = {}
        for var, (start, stop) in self._var_layout:
            vars[var] = data[start:stop]
        return vars

    ###########################################################################
    # constants handling
    ###########################################################################

    @property
    def constants(self):
        """
        Constant values for variables.
        restore_constants must be called after every update to ensure that constants have
        their requested value.
        """
        return self._constants.copy()

    @constants.setter
    def constants(self, value):
        self._constants = dict(value)

        # build constant selector and zero array for nullifying gradient
        idxs = [np.arange(*self.extents_of_var(var)) for var in self._constants.iterkeys()]
        sel = reduce(lambda a, b: np.concatenate((a, b)), idxs, [])
        sel = np.sort(sel)
        self._constants_selector = post(sel)
        self._constants_zeros = xp.zeros((len(sel),))

    def restore_constants(self):
        """
        Ensures that all constant variables have their required values.
        """
        for var, value in self._constants.iteritems():
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
        if len(self._constants) == 0:
            return grad
        grad[self._constants_selector] = self._constants_zeros
        return grad

    ###########################################################################
    # expose variables as attributes and elements
    ###########################################################################

    def __contains__(self, key):
        """Numeric variables are exposed as dict elements."""
        return key in self._num_var_slices

    def __getitem__(self, key):
        """Numeric variables are exposed as dict elements."""
        return self.num(key)

    def __setitem__(self, key, value):
        """Numeric variables are exposed as dict elements."""
        self._num_var_slices[key][:] = value

    def __getattr__(self, item):
        """Symbolic variables are exposed as attributes."""
        return self.sym(item)

    def all_vars(self):
        """List of all variable names."""
        return list(self._num_var_slices.keys())

    ###########################################################################
    # legacy interface
    ###########################################################################

    @property
    def vars(self):
        """
        Dicitionary of variable names and their corresponding symbolic variables.
        Deprecated: use sym_vars instead.
        """
        return self.sym_vars

    @property
    def data(self):
        """
        Numerical data values of this ParameterSet
        Deprecated: use num_data instead.
        """
        return self.num_data

    @property
    def flat(self):
        """
        Symbolic Theano variable corresponding to the values of this ParameterSet.
        Deprecated: use sym_data instead.
        """
        return self._sym_data





