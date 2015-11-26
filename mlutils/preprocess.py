from __future__ import division
import numpy as np
from mlutils.misc import combine_sample_steps, divide_sample_steps


def pca_white(data, n_components=None, return_axes=False):
    """
    Performs PCA whitening of the data.
    For every component the mean over the samples is zero and the variance is one.
    The covariance matrix of the components is diagonal.

    :param data: data[feature, smpl] - input data
    :param n_components: If specified, limits the number of principal components to keep.
    :param return_axes: If True, this function additionaly returns the PCA variances, corresponding axes and means.
    :return: if return_axes=False: whitened[comp, smpl] - PCA whitened data.
             if return_axes=True: (whitened[comp, smpl], variances[comp], axes[dim, comp], means[dim]).
    """
    data = np.asarray(data)
    n_samples = data.shape[1]

    # center data on mean
    means = np.mean(data, axis=1)
    centered = data - means[:, np.newaxis]

    # calculate principal axes and transform data into that coordinate system
    cov = np.dot(centered, centered.T) / n_samples
    variances, axes = np.linalg.eigh(cov)
    sort_idx = np.argsort(variances)[::-1]
    variances = variances[sort_idx]
    axes = axes[:, sort_idx]
    if n_components is not None:
        variances = variances[0:n_components]
        axes = axes[:, 0:n_components]
    pcaed = np.dot(axes.T, centered)

    # scale axes so that each has unit variance
    whitened = np.dot(np.diag(np.sqrt(1.0 / variances)), pcaed)

    if return_axes:
        return whitened, variances, axes, means
    else:
        return whitened

def pca_white_inverse(whitened, variances, axes, means):
    """
    Restores original data from PCA whitened data.
    :param whitened: whitened[comp, smpl] - PCA whitened data.
    :param variances: variances[comp] - variances of PCA components
    :param axes: axes[dim, comp] - PCA axes
    :param means: means[dim] - data means
    :return: data[feature, smpl] - reconstructed data
    """
    whitened = np.asarray(whitened)
    variances = np.asarray(variances)
    axes = np.asarray(axes)
    means = np.asarray(means)

    # reverse unit variance
    pcaed = np.dot(np.diag(np.sqrt(variances)), whitened)

    # restore original coordinate system
    centered = np.dot(axes, pcaed)

    # restore mean
    data = centered + means[:, np.newaxis]

    return data


def zca(data):
    """
    Performs zero phase component analysis (ZCA) of the data.
    :param data: data[feature, smpl] - input data
    :return: zcaed[feature, smpl] - ZCA whitened data.
             For every feature the mean over the samples is zero and the variance is one.
             The covariance matrix is diagonal.
             Furhtermore zcaed is most similar to data in the least square sense, i.e. (zcaed - data)**2 is minimized
             with the constraint that above properties are satisfied.
    """
    # get PCA whitened data
    whitened, variances, axes = pca_white(data, return_axes=True)

    # rotate back into original coordinate system
    zcaed = np.dot(axes, whitened)

    return zcaed


def for_step_data(func):
    """
    Builds a preprocessing function for step data, i.e. data of the form data[feature, step, smpl].
    The built function expects the number of steps for each sample as first argument and the data as second argument.
    It returns the preprocesd data in the same shape as the input data.
    :param func: the preprocessing function
    :return: preprocessing_func(n_steps[smpl], data[feature, step, smpl])
    """
    def sd_func(n_steps, data, *args, **kwargs):
        data_c = combine_sample_steps(n_steps, data)
        ret = func(data_c, *args, **kwargs)
        if isinstance(ret, tuple):
            data_pc = ret[0]
        else:
            data_pc = ret
        data_p = divide_sample_steps(n_steps, data_pc)
        if isinstance(ret, tuple):
            return (data_p,) + ret[1:]
        else:
            return data_p
    return sd_func


def scale(data, min=0.0, max=1.0):
    """
    :param data: data[feature, smpl] - input data
    :param min: minimum value of the scaled data for each feature
    :param max: maximum value of the scaled data for each feature
    :return: data with each feature scaled to lie within [min, max]
    """
    # Compute max and min within each feature
    maxs = np.max(data, axis=1)
    mins = np.min(data, axis=1)
    # Scale data to be between 0 and 1 (transpose needed to use numpys broad-
    # casting for subtraction
    data_s = (data - mins[:, np.newaxis])/(maxs - mins)[:, np.newaxis]
    # Scale to given min and max
    return (max - min) * data_s + min
