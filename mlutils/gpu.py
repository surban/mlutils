# mostly copied from Breze
import os
import sys
import theano
import theano.sandbox.cuda
import gnumpy
import numpy as np

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
        import theano.sandbox.cuda
        if not theano.sandbox.cuda.cuda_available:
            GPU = False

if GPU:
    import theano.misc.gnumpy_utils as gput
    print "Device: GPU"
else:
    print "Device: CPU"


def floatx(x):
    """Converts the numpy array to use Theano's float type"""
    return np.asarray(x, dtype=theano.config.floatX)


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
    # expr = T.cast(expr, 'float32')
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
    lst_flat = [gput.garray_to_cudandarray(i)
                if not isinstance(i, theano.sandbox.cuda.CudaNdarray) else i
                for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def cudandarray_to_garray_nested(lst):
    lst_flat = flatten(lst)
    lst_flat = [gput.cudandarray_to_garray(i, copyif=True) for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def gnumpy_to_ndarray_wrap(f):
    """Wrap a function that accepts and returns CudaNdArrays to accept and
    return gnumpy arrays."""

    def inner(*args):
        args = garray_to_cudandarray_nested(args)
        res = f(*args)
        # print "res:", res
        # return res
        # return None
        if isinstance(res, list):
            res = cudandarray_to_garray_nested(res)
        else:
            # TODO: check for CudaNdArray instance instead
            if not isinstance(res, (float, np.ndarray)):
                res = gput.cudandarray_to_garray(res, copyif=True)
        return res

    return inner


def var_exp_for_gpu(variables, exprs, outputs=True):
    """Given variables and theano expressions built from these variables,
    return variables and expressions of the same form that are tailored
    towards GPU usage."""

    # Here is the outline of this function.
    #
    # (1) For each CPU tensor from theano.tensor create a corresponding GPU
    #     tensor from theano.sandbox.cuda,
    # (2) replace these in all expressions,
    # (3) replace the output expressions with GPU expressions so no
    #     auto-conversion to numpy is done.
    #
    # Since variables and expressions might be nested, we need to flatten
    # them first and unflatten the results.

    # Stage (1)
    variables_flat = flatten(variables)
    gpu_var_flat = []
    gpu_variable_subs = {}
    for var in variables_flat:
        if var in gpu_variable_subs:
            gpu_var = gpu_variable_subs[var]
        else:
            gpu_var = cpu_tensor_to_gpu(var)
            gpu_variable_subs[var] = gpu_var
        gpu_var_flat.append(gpu_var)
    gpu_variables = unflatten(variables, gpu_var_flat)

    # Loop for stage (2) and (3):
    exprs_flat = flatten(exprs)
    gpu_exprs_flat = []
    for expr in exprs_flat:
        # (2)
        for v, gv in zip(variables_flat, gpu_var_flat):
            expr = theano.clone(expr, {v: gv})
        # (3)
        if outputs:
            expr = cpu_expr_to_gpu(expr)
        gpu_exprs_flat.append(expr)

    gpu_exprs = unflatten(exprs, gpu_exprs_flat)

    return gpu_variables, gpu_exprs


def function(inputs, outputs, name=None, plot_graph=False, *args, **kwargs):
    import theano
    if GPU:
        gpu_inputs, gpu_outputs = var_exp_for_gpu(inputs, outputs)
        f = theano.function(gpu_inputs, gpu_outputs, name=name, *args, **kwargs)
        if plot_graph:
            import theano.printing
            theano.printing.pydotprint(f, outfile="%s.png" % name,
                                       var_with_name_simple=True)
        f = gnumpy_to_ndarray_wrap(f)
    else:
        f = theano.function(inputs, outputs, name=name, *args, **kwargs)
        if plot_graph:
            import theano.printing
            theano.printing.pydotprint(f, outfile="%s.png" % name,
                                       var_with_name_simple=True)
    return f


def gather(x):
    """Copys array from GPU if running on GPU"""
    if GPU:
        return gnumpy.as_numpy_array(x)
    else:
        return x


def post(x):
    """Copys array to GPU if running on GPU"""
    if GPU:
        return gnumpy.as_garray(x)
    else:
        return floatx(x)
