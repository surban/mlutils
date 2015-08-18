# -*- coding: utf-8 -*-
# mostly taken from Breze

import os
import sys
import collections
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
import theano.misc.gnumpy_utils as gput


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


def flatten(nested):
    """Flatten nested tuples and/or lists into a flat list."""
    if isinstance(nested, (tuple, list)):
        flat = []
        for elem in nested:
            flat.extend(flatten(elem))
        return flat
    else:
        return [nested]


def unflatten(tmpl, flat):
    """Nest the items in flat into the shape of tmpl."""
    def unflatten_recursive(tmpl, flat):
        if isinstance(tmpl, (tuple, list)):
            nested = []
            for sub_tmpl in tmpl:
                sub_nested, flat = unflatten_recursive(sub_tmpl, flat)
                nested.append(sub_nested)
            if isinstance(tmpl, tuple):
                nested = tuple(nested)
            return nested, flat
        else:
            return flat[0], flat[1:]

    nested, _ = unflatten_recursive(tmpl, flat)
    return nested


def theano_function_with_nested_exprs(variables, exprs, *args, **kwargs):
    """Creates and returns a theano.function that takes values for `variables`
    as arguments, where `variables` may contain nested lists and/or tuples,
    and returns values for `exprs`, where again `exprs` may contain nested
    lists and/or tuples.

    All other arguments are passed to theano.function without modification."""

    flat_variables = flatten(variables)
    flat_exprs = flatten(exprs)

    flat_function = theano.function(
        flat_variables, flat_exprs, *args, **kwargs)

    def wrapper(*fargs):
        flat_fargs = flatten(fargs)
        flat_result = flat_function(*flat_fargs)
        result = unflatten(exprs, flat_result)
        return result

    # Expose this to the outside so that fields of theano can be accessed, eg
    # for debug or graph information.
    wrapper.flat_function = flat_function

    return wrapper


def cpu_tensor_to_gpu(tensor):
    """Given a tensor for the CPU return a tensor of the same type and name for
    the GPU."""
    name = '%s-gpu' % tensor.name
    if tensor.ndim == 0:
        result = theano.sandbox.cuda.fscalar(name)
    elif tensor.ndim == 1:
        result = theano.sandbox.cuda.fvector(name)
    elif tensor.ndim == 2:
        result = theano.sandbox.cuda.fmatrix(name)
    elif tensor.ndim == 3:
        result = theano.sandbox.cuda.ftensor3(name)
    elif tensor.ndim == 4:
        result = theano.sandbox.cuda.ftensor4(name)
    else:
        raise ValueError('only up to dimension 4')

    return result


def cpu_tensor_to_gpu_nested(inpts, cache=None):
    """Given a list (of lists of...) CPU tensor variables return as list of the
    same types of corresponding GPU tensor varaibles.

    Also return a dictionary containing all substitutions done. This can
    be provided to future calls to not make conversions multiple times.
    """
    if cache is None:
        cache = {}
    inpts_flat = flatten(inpts)
    inpts_flat_conv = []
    for inpt in inpts_flat:
        if inpt in cache:
            item = cache[inpt]
        else:
            item = cpu_tensor_to_gpu(inpt)
            cache[inpt] = item
        inpts_flat_conv.append(item)

    return unflatten(inpts, inpts_flat_conv), cache


def cpu_expr_to_gpu(expr, unsafe=False):
    """Given a CPU expr return the same expression for the GPU.

    If unsafe is set to True, subsequent function calls evaluating the
    expression might return arrays pointing at the same memory region.
    """
    expr = T.cast(expr, 'float32')
    return theano.Out(theano.sandbox.cuda.basic_ops.gpu_from_host(expr),
                      borrow=unsafe)


def cpu_expr_to_gpu_nested(inpts, unsafe=False):
    """Given a list (of lists of...) expressions, return expressions for the
    GPU.

    If unsafe is set to True, subsequent function calls evaluating the
    expression might return arrays pointing at the same memory region.
    """
    inpts_flat = flatten(inpts)
    inpts_flat = [cpu_expr_to_gpu(i, unsafe) for i in inpts_flat]
    return unflatten(inpts, inpts_flat)


def garray_to_cudandarray_nested(lst):
    lst_flat = flatten(lst)
    lst_flat = [gput.garray_to_cudandarray(i) for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def cudandarray_to_garray_nested(lst):
    lst_flat = flatten(lst)
    lst_flat = [gput.cudandarray_to_garray(i) for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def gnumpy_func_wrap(f):
    """Wrap a function that accepts and returns CudaNdArrays to accept and
    return gnumpy arrays."""
    def inner(*args):
        args = garray_to_cudandarray_nested(args)
        res = f(*args)
        if isinstance(res, list):
            res = cudandarray_to_garray_nested(res)
        else:
            # TODO: check for CudaNdArray instance instead
            if not isinstance(res, (float, np.ndarray)):
                res = gput.cudandarray_to_garray(res)
        return res
    return inner


def lookup(what, where, default=None):
    """Return ``where.what`` if what is a string, otherwise what. If not found
    return ``default``."""
    if isinstance(what, (str, unicode)):
        res = getattr(where, what, default)
    else:
        res = what
    return res


def lookup_some_key(what, where, default=None):
    """Given a list of keys ``what``, return the first of those to which there
    is an item in ``where``.

    If nothing is found, return ``default``.
    """
    for w in what:
        try:
            return where[w]
        except KeyError:
            pass
    return default


def opt_from_model(model, fargs, args, opt_klass, opt_kwargs):
    """Return an optimizer object given a model and an optimizer specification.
    """
    d_loss_d_pars = T.grad(model.exprs['loss'], model.parameters.flat)
    f = model.function(fargs, 'loss', explicit_pars=True)
    fprime = model.function(fargs, d_loss_d_pars, explicit_pars=True)
    opt = opt_klass(model.parameters.data, f, fprime, args=args, **opt_kwargs)
    return opt


def theano_expr_bfs(expr):
    """Generator function to walk a Theano expression graph in breadth first."""
    stack = [expr]
    while True:
        if not stack:
            break
        expr = stack.pop()
        stack += expr.owner.inputs if hasattr(expr.owner, 'inputs') else []
        yield expr


def tell_deterministic(expr):
    """Return True iff no random number generator is in the expression graph."""
    return all(not hasattr(i, 'rng') for i in theano_expr_bfs(expr))


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
            self.data = gnumpy.zeros(self.n_pars)
            self.flat = theano.sandbox.cuda.fvector('parameters')
        else:
            self.data = np.empty(self.n_pars).astype(theano.config.floatX)
            self.flat = T.vector('parameters')

        self.flat.tag.test_value = self.data

        # Go through parameters and assign space and variable.
        self.views = {}
        n_used = 0 	# Number of used parameters.

        for (key, shape), size in zip(kwargs.items(), sizes):
            # Make sure the key is legit -- that it does not overwrite
            # anything.
            if hasattr(self, key):
                raise ValueError("%s is an illegal name for a variable")

            # Get the region from the big flat array.
            region = self.data[n_used:n_used + size]
            # Then shape it correctly and make it accessible from the outside.
            region = region.reshape(shape)
            self.views[key] = region

            # Get the right variable as a subtensor.
            var = self.flat[n_used:n_used + size].reshape(shape)
            var.name = key
            setattr(self, key, var)

            n_used += size

    @property
    def vars(self):
        """All Theano vectors can be accessed by with their identifier as key
        in this dictionary."""
        return {k: getattr(self, k) for k in self.views.iterkeys()}

    def __contains__(self, key):
        return key in self.views

    def __getitem__(self, key):
        return self.views[key]

    def __setitem__(self, key, value):
        self.views[key][:] = value


