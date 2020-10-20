# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains sparsity operator classes.

"""

import numpy as np
import pywt
import scipy

from .utils import (flatten_swtn, unflatten_swtn,
                    flatten_wave, unflatten_wave)


class pyWavelet:
    """ The 3D wavelet transform class from pyWavelets package"""

    def __init__(self, wavelet_name, nb_scale=4, undecimated=False,
                 mode='zero'):
        """
        Initialize the 'pyWavelet3' class.

        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        undecimated: bool, default False
            enable the use of undecimated wavelet transform.
        mode : str or tuple of str, optional
            Signal extension mode, see :ref:`Modes <ref-modes>`. This can also
            be a tuple containing a mode to apply along each axis in ``axes``.
        """
        if wavelet_name not in pywt.wavelist():
            raise ValueError("Unknown transformation '{}'".format(wavelet_name))
        self.pywt_transform = pywt.Wavelet(wavelet_name)
        self.nb_scale = nb_scale
        self.undecimated = undecimated
        self.unflatten = unflatten_swtn if undecimated else unflatten_wave
        self.flatten = flatten_swtn if undecimated else flatten_wave
        self.coeffs = None
        self.coeffs_shape = None
        self.mode = mode

    def get_coeff(self):
        """
        Return the wavelet coefficients

        Return:
        -------
        The values of the wavelet coefficients
        """
        return self.coeffs

    def set_coeff(self, coeffs):
        """ Set wavelets decomposition coefficients values"""
        self.coeffs = coeffs  # XXX: TODO: add some checks

    def op(self, data):
        """
        Define the wavelet operator.
        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: np.ndarray(m', n') or np.ndarray(m', n', p')
            input 2D or 3D data array.

        Returns
        -------
        coeffs: np.ndarray
            the wavelet coefficients.
        """
        if self.undecimated:
            coeffs_dict = pywt.swtn(data, self.pywt_transform,
                                    level=self.nb_scale)
            coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return coeffs
        else:
            coeffs_dict = pywt.wavedecn(data,
                                        self.pywt_transform,
                                        level=self.nb_scale,
                                        mode=self.mode)
            self.coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return self.coeffs

    def adj_op(self, coeffs):
        """
        Define the wavelet adjoint operator.
        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: np.ndarray
            the wavelet coefficients.

        Returns
        -------
        data: np.ndarray((m, n)) or np.ndarray((m, n, p))
            the 2D or 3D reconstructed data.
        """
        self.coeffs = coeffs
        if self.undecimated:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.iswtn(coeffs_dict,
                              self.pywt_transform)
        else:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.waverecn(
                coeffs=coeffs_dict,
                wavelet=self.pywt_transform,
                mode=self.mode)
        return data

    def l2norm(self, shape):
        """
        Compute the L2 norm.

        Parameters
        ----------
        shape: tuple of int
            the 2D or 3D data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        np.put(fake_data, fake_data.size // 2, 1)

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


class HOTV:
    """ The HOTV computation class for 2D image decomposition

    .. note:: At the moment, it assumed that the image is square
    """

    def __init__(self, img_shape, order=1):
        """
        Initialize the 'HOTV' class.

        Parameters
        ----------
        img_shape: tuple of int
            image dimensions
        order: int, optional
            order of the differential operator used for the HOTV computation
        """
        assert (img_shape[0] == img_shape[1])

        self.img_size = img_shape[0]
        self.filter = np.zeros((order + 1, 1))
        for k in range(order + 1):
            self.filter[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

        offsets_x = np.arange(order + 1)
        offsets_y = self.img_size * np.arange(order + 1)
        shape = (self.img_size ** 2,) * 2
        sparse_mat_x = scipy.sparse.diags(self.filter,
                                          offsets=offsets_x, shape=shape)
        sparse_mat_y = scipy.sparse.diags(self.filter,
                                          offsets=offsets_y, shape=shape)

        self.op_matrix = scipy.sparse.vstack([sparse_mat_x, sparse_mat_y])

    def op(self, data):
        """
        Define the HOTV operator.
        This method returns the input data convolved with the HOTV filter.

        Parameters
        ----------
        data: np.ndarray((m', m'))
            input 2D data array.

        Returns
        -------
        coeffs: np.ndarray((2*m'*m'))
            the variation values.
        """
        return self.op_matrix * (data.flatten())

    def adj_op(self, coeffs):
        """
        Define the HOTV adjoint operator.
        This method returns the adjoint of HOTV computed image.

        Parameters
        ----------
        coeffs: np.ndarray((2*m'*m'))
            the HOTV coefficients.

        Returns
        -------
        data: np.ndarray((m', m'))
            the reconstructed data.
        """
        return np.reshape(self.op_matrix.T * coeffs,
                          (self.img_size, self.img_size))

    def l2norm(self, shape):
        """
        Compute the L2 norm.

        Parameters
        ----------
        shape: tuple
            the 2D data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        np.put(fake_data, fake_data.size // 2, 1)

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


class HOTV_3D:
    """
    The HOTV computation class for 3D image decomposition

    .. note:: At the moment, assumed that the image is square in x-y directions
    """

    def __init__(self, img_shape, nb_slices, order=1):
        """
        Initialize the 'HOTV' class.

        Parameters
        ----------
        img_shape: tuple of int
            image dimensions (assuming that the image is square)
        nb_slices: int
            number of slices in the 3D reconstructed image
        order: int, default is 1
            order of the differential operator used for the HOTV computation
        """
        assert (img_shape[0] == img_shape[1])

        self.img_size = img_shape[0]
        self.nb_slices = nb_slices
        self.filter = np.zeros((order + 1, 1))
        for k in range(order + 1):
            self.filter[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

        offsets_x = np.arange(order + 1)
        offsets_y = self.nb_slices * np.arange(order + 1)
        offsets_z = (self.img_size * self.nb_slices) * np.arange(order + 1)
        shape = ((self.img_size ** 2) * self.nb_slices,) * 2
        sparse_mat_x = scipy.sparse.diags(self.filter,
                                          offsets=offsets_x, shape=shape)
        sparse_mat_y = scipy.sparse.diags(self.filter,
                                          offsets=offsets_y, shape=shape)
        sparse_mat_z = scipy.sparse.diags(self.filter,
                                          offsets=offsets_z, shape=shape)

        self.op_matrix = scipy.sparse.vstack(
            [sparse_mat_x, sparse_mat_y, sparse_mat_z])

    def op(self, data):
        """
        Define the HOTV operator.
        This method returns the input data convolved with the HOTV filter.

        Parameters
        ----------
        data: np.ndarray((m', m', p'))
            input 3D data array.

        Returns
        -------
        coeffs: np.ndarray((3*m'*m'*p'))
            the variation values.
        """
        return self.op_matrix * (data.flatten())

    def adj_op(self, coeffs):
        """
        Define the HOTV adjoint operator.
        This method returns the adjoint of HOTV computed image.

        Parameters
        ----------
        coeffs: np.ndarray((3*m'*m'*p'))
            the HOTV coefficients.

        Returns
        -------
        data: np.ndarray((m', m', p'))
            the reconstructed data.
        """
        return np.reshape(self.op_matrix.T * coeffs,
                          (self.img_size, self.img_size, self.nb_slices))

    def l2norm(self, shape):
        """
        Compute the L2 norm.

        Parameters
        ----------
        shape: tuple
            the 3D data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        np.put(fake_data, fake_data.size // 2, 1)

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)
