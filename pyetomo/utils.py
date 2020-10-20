# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains useful methods for tomographic applications.
"""
import numpy as np
import scipy.fftpack as pfft


def with_metaclass(meta, *bases):
    """
    Function from jinja2/_compat.py.

    License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """

    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)

    return metaclass("temporary_class", None, {})


def monkeypatch(klass, methodname=None):
    """ Decorator extending class with the decorated callable.

    >>> class A:
    ...     pass
    >>> @monkeypatch(A)
    ... def meth(self):
    ...     return 12
    ...
    >>> a = A()
    >>> a.meth()
    12
    >>> @monkeypatch(A, 'foo')
    ... def meth(self):
    ...     return 12
    ...
    >>> a.foo()
    12

    Parameters
    ----------
    klass: class object
        the class to be decorated.
    methodname: str, default None
        the name of the decorated method. If None, use the function name.

    Returns
    -------
    decorator: callable
        the decorator.
    """

    def decorator(func):
        try:
            name = methodname or func.__name__
        except AttributeError:
            raise AttributeError(
                "{0} has no __name__ attribute: you should provide an "
                "explicit 'methodname'".format(func))
        setattr(klass, name, func)
        return func

    return decorator


def flatten(x):
    """ Flattens list an array.

    Parameters
    ----------
    x: list of ndarray or ndarray
        the input dataset.

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of uplet
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape = [x[0].shape]
    for data in x[1:]:
        y = np.concatenate((y, data.flatten()))
        shape.append(data.shape)

    return y, shape


def unflatten(y, shape):
    """ Unflattens a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of uplet
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    offset = 0
    x = []
    for size in shape:
        start = offset
        stop = offset + np.prod(size)
        offset = stop
        x.append(y[start: stop].reshape(size))

    return x


def flatten_swtn(x):
    """ Flattens list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = []
    shape_dict = []
    for i in range(len(x)):
        dict_lvl = {}
        for key in x[i].keys():
            dict_lvl[key] = x[i][key].shape
            y = np.concatenate((y, x[i][key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_swtn(y, shape):
    """ Unflattens a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    x = []
    offset = 0
    for i in range(len(shape)):
        sublevel = {}
        for key in shape[i].keys():
            start = offset
            stop = offset + np.prod(shape[i][key])
            offset = stop
            sublevel[key] = y[start: stop].reshape(shape[i][key])
        x.append(sublevel)
    return x


def flatten_wave(x):
    """ Flattens list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    for x_i in x[1:]:
        dict_lvl = []
        for key in x_i.keys():
            dict_lvl.append((key, x_i[key].shape))
            y = np.concatenate((y, x_i[key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_wave(y, shape):
    """ Unflattens a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    for shape_i in shape[1:]:
        sublevel = {}
        for key, value in shape_i:
            start = offset
            stop = offset + np.prod(value)
            offset = stop
            sublevel[key] = y[start: stop].reshape(value)
        x.append(sublevel)
    return x


def generate_locations_etomo_3D(size_x, size_z, angles):
    """
    This function generates the list of the samples' coordinate in the k-space.

    Parameters
    ----------
    size_x: int
        image size along the x-axis
    size_z: int
        image size along the z-axis (rotation axis)
    angles: np.ndarray((q))
        array countaining the acquisition angles

    Returns
    -------
    samples: np.ndarray((int(np.floor(np.sqrt(2) * size_x)) * size_z * q, 3))
        Fourier space locations generated from the given angles and data image
        size

    """
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    rho = np.tile(np.linspace(-0.5, 0.5, diag_x, endpoint=False), size_z)
    k_z = np.tile(np.reshape(np.linspace(-0.5, 0.5, size_z, endpoint=False),
                             (size_z, 1)), (1, diag_x)).flatten()
    for t, angle in enumerate(angles):
        sample = np.zeros((diag_x * size_z, 3))
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        sample[:, 2] = k_z
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


def generate_locations_etomo_2D_SL(size_x, angles):
    """
    This function generates the list of the samples' coordinates in the k-space.

    Parameters
    ----------
    size_x: int
        image size along the x-axis
    angles: np.ndarray((q))
        array countaining the acquisition angles

    Returns
    -------
    samples: np.ndarray((size_x * q, 2))
        Fourier space locations generated from the given angles and data image
        size
    """
    rho = np.linspace(-0.5, 0.5, size_x, endpoint=False)
    for t, angle in enumerate(angles):
        sample = np.zeros((size_x, 2))
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


def generate_locations_etomo_2D(size_x, angles):
    """
    This function generates the list of the samples' coordinates in the k-space.

    Parameters
    ----------
    size_x: int
        image size along the x-axis
    angles: np.ndarray((q))
        array countaining the acquisition angles

    Returns
    -------
    samples: np.ndarray((size_x * q, 2))
        Fourier space locations generated from the given angles and data image
        size
   """
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    rho = np.linspace(-0.5, 0.5, diag_x, endpoint=False)
    for t, angle in enumerate(angles):
        sample = np.zeros((diag_x, 2))
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


# def generate_mask_etomo_2D(size_x, angles):
#     """This function generates the mask locations of the sample.
#
#     Parameters
#     ----------
#     size_x: int
#         image size along the x-axis
#
#     angles: np.ndarray((q))
#         array countaining the acquisition angles
#
#     Returns
#     -------
#     mask: np.ndarray(())
#         Mask locations of the Fourier space data corresponding to the
#         acquired angles
#    """
#     mask = np.zeros((size_x, size_x))
#     Xx = []
#     Yy = []
#     for angle in angles:
#         rho = size_x * np.linspace(-0.5, 0.5, size_x, endpoint=False)
#         X = rho * np.cos(angle * np.pi / 180) + size_x / 2
#         Y = rho * np.sin(angle * np.pi / 180) + size_x / 2
#         X_mask = np.where(X >= size_x)
#         Y_mask = np.where(Y >= size_x)
#         X = np.delete(X, X_mask, 0)
#         X = np.delete(X, Y_mask, 0)
#         Y = np.delete(X, X_mask, 0) # BUG TO FIX ?
#         Y = np.delete(X, Y_mask, 0)
#         Xx = np.concatenate([Xx, X])
#         Yy = np.concatenate([Yy, Y])
#     mask[Xx.astype(int), Yy.astype(int)] = 1
#     return mask


def generate_kspace_etomo_2D(sinogram):
    """
    This function generates the list of the kspace observations (with zero-padding).

    Parameters
    ----------
    sinogram: np.ndarray((q, m))
        sinogram with size nb_angles and size_x (m)

    Returns
    -------
    kspace_obs: np.ndarray((q*int(m*sqrt(2)))
        Fourier space values from the given sinogram
    """
    nb_angles, size_x = sinogram.shape
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    jmin = int(np.floor((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    jmax = -int(np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    sinograms_zp = np.zeros((nb_angles, diag_x))
    sinograms_zp[:, jmin:jmax] = sinogram

    # nb_angles, size_x = sinogram.shape
    # diag_x = int(np.floor(np.sqrt(2) * size_x))
    # sinograms_zp = np.zeros((nb_angles, diag_x))
    # sinograms_zp[:, (diag_x - size_x) / 2):-int(
    # np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))] = sinogram

    ft_sinogram = []
    for t in range(sinogram.shape[0]):
        ft_sinogram.append(pfft.fftshift(pfft.fft(pfft.ifftshift(
            sinograms_zp[t].astype("complex128")))))

    ft_sinogram = np.asarray(ft_sinogram).flatten()
    kspace_obs = ft_sinogram.flatten()
    return kspace_obs


def generate_kspace_etomo_2D_SL(sinogram):
    """
    This function generates the list of the kspace observations (without zero-padding).

    Parameters
    ----------
    sinogram: np.ndarray((q, m))
        sinogram with size nb_angles and size_x (m)

    Returns
    -------
    kspace_obs: np.ndarray((q*m))
        Fourier space values from the given sinogram
    """
    # nb_angles, Nx = sinogram.shape
    # sinograms_zp = np.zeros((nb_angles, int(np.floor(np.sqrt(2) * Nx))))
    # sinograms_zp[:, int(np.floor((np.floor(np.sqrt(2) * Nx) - Nx) / 2)):-int(
    #     np.ceil((np.floor(np.sqrt(2) * Nx) - Nx) / 2))] = sinogram

    ft_sinogram = []
    for t in range(sinogram.shape[0]):
        ft_sinogram.append(pfft.fftshift(pfft.fft(pfft.ifftshift(
            sinogram[t].astype("complex128")))))

    ft_sinogram = np.asarray(ft_sinogram).flatten()
    kspace_obs = ft_sinogram.flatten()
    return kspace_obs


def generate_kspace_etomo_3D(sinograms):
    """
    This function generates the list of the kspace observations (with zero-padding).

    Parameters
    ----------
    sinogram: np.ndarray((q, m, p))
        sinogram with size nb_angles and size_x, size_z (m, p)

    Returns
    -------
    kspace_obs: np.ndarray((q*int(m*sqrt(2)*p))
        Fourier space values from the given sinogram
    """
    nb_angles, size_x, size_z = sinograms.shape
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    jmin = int(np.floor((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    jmax = -int(np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    sinograms_zp = np.zeros((nb_angles, diag_x, size_z))
    sinograms_zp[:, jmin:jmax, :] = sinograms

    ft_sinograms = []
    for t in range(nb_angles):
        ft_sinograms.append(pfft.fftshift(pfft.fft2(pfft.ifftshift(
            sinograms_zp[t].astype("complex128")))).T.flatten())

    ft_sinograms = np.asarray(ft_sinograms).flatten()
    kspace_obs = ft_sinograms.flatten()
    return kspace_obs
