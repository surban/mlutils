import theano.tensor as T
import numpy as np


def clog(re, im):
    """
    Compute complex logarithm given a (tensor of) complex numbers(s) given by
    their real and imaginary part
    :param re: real part of the complex number(s)
    :param im: imaginary part of the complex number(s)
    :return: Log(re + im)
    """
    # z = a + ib -> Log z = ln(sqrt(a^2+b^2)) + i*atan2(b,a)
    log_re = 0.5 * T.log(re ** 2 + im ** 2)
    log_im = T.arctan2(im, re)
    return log_re, log_im


def cexp(re, im):
    """
    Compute complex logarithm given a (tensor of) complex numbers(s) given by
    their real and imaginary part
    :param re: real part of the complex number(s)
    :param im: imaginary part of the complex number(s)
    :return: Exp(re + im)
    """
    # z = a + bi -> Exp z = exp(a)*(cos(b) + isin(b))
    exp_re = T.exp(re) * T.cos(im)
    exp_im = T.exp(re) * T.sin(im)
    return exp_re, exp_im


def cdot(a_re, a_im, b_re, b_im):
    """
    Compute dot product of two arrays of complex numbers given by their real
    and imaginary parts
    :param a_re: real part of the first complex number(s)
    :param a_im: imaginary part of the first complex number(s)
    :param b_re: real part of the second complex number(s)
    :param b_im: imaginary part of the first complex number(s)
    :return: dot product of a_re + a_im and b_re + b_im
    """
    # (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
    # where a_re = A, a_im = B, b_re = C, b_im = D
    dot_re = T.dot(a_re, b_re) - T.dot(a_im, b_im)
    dot_im = T.dot(a_im, b_re) + T.dot(a_re, b_im)
    return dot_re, dot_im


def np_cdot(a_re, a_im, b_re, b_im):
    """ Numpy equivalent of cdot """
    dot_re = np.dot(a_re, b_re) - np.dot(a_im, b_im)
    dot_im = np.dot(a_im, b_re) + np.dot(a_re, b_im)
    return dot_re, dot_im


def cmul(a_re, a_im, b_re, b_im):
    """
    Compute the product of two complex numbers given bei their real and
    imaginary part
    :param a_re: real part of the first complex number
    :param a_im: imaginary part of the first complex number
    :param b_re: real part of the second complex number(s)
    :param b_im: imaginary part of the first complex number(s)
    :return: product of a_re + a_im and b_re + b_im
    """
    # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
    # where a_re = a, a_im = b, b_re = c, b_im = d
    mul_re = a_re * b_re - a_im * b_im
    mul_im = a_im * b_re + a_re * b_im
    return mul_re, mul_im
