# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
2D and 3D Non-Uniform FFT (NUFFT) operators and their adjoints.
"""
import pynufft


class FourierBase:
    """ Base Fourier transform operator class"""

    def op(self, img):
        """ This method calculates a Fourier transform"""
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, x):
        """ This method calculates the adjoint Fourier transform of a real or
        complex sequence"""
        raise NotImplementedError("'adj_op' is an abstract method.")


class NUFFT3(FourierBase):
    """ Standard 3D non cartesian Fast Fourier Transform class

    Attributes
    ----------
    samples: np.ndarray((m'*n'*p', 3))
        samples' coordinates in the Fourier domain.
    shape: tuple of int
        shape of the final reconstructed image (m, n, p) (not necessarily a
        square matrix).
    """

    def __init__(self, samples, shape):
        """ Initialize the 'NUFFT3' class.

        Parameters
        ----------
        samples: np.ndarray((m'*n'*p', 3))
            samples' coordinates in the Fourier domain.
        shape: tuple of int
            shape of the final reconstructed image (m, n, p) (not necessarily
            a square matrix).
        """
        self.plan = pynufft.NUFFT()
        shape_fourier = []
        for dim_size in shape:
            shape_fourier.append(int(2 * dim_size))
        self.plan.plan(samples, tuple(shape), tuple(shape_fourier), (5, 5, 5))
        self.shape = shape
        self.samples = samples

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray((m, n, p))
            input 3D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray((m*n*p))
            masked Fourier transform of the input image.
        """
        self.plan.forward(img)
        return self.plan.forward(img)

    def adj_op(self, x):
        """ This method calculates the adjoint non-cartesian Fourier transform
        of a 3-D array.

        Parameters
        ----------
        x: np.ndarray((m'*n'*p'))
            masked non-cartesian Fourier transform 3D data.

        Returns
        -------
        img: np.ndarray((m, n, p))
            adjoint 3D discrete Fourier transform of the input coefficients.
        """
        return self.plan.adjoint(x)


class NUFFT2(FourierBase):
    """ Standard 2D non catesian Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray((m'*n, 3))
        samples' coordinates in the Fourier domain.
    shape: tuple of int
        shape of the final reconstructed image (m, n) (not necessarily a
        square matrix).
    """

    def __init__(self, samples, shape):
        """ Initialize the 'NUFFT2' class.

        Parameters
        ----------
        samples: np.ndarray((m'*n', 3))
            samples' coordinates in the Fourier domain.
        shape: tuple of int
            shape of the final reconstructed image (m, n) (not necessarily a
            square matrix).
        """
        self.plan = pynufft.NUFFT()
        shape_fourier = []
        for dim_size in shape:
            shape_fourier.append(int(2 * dim_size))
        self.plan.plan(samples, tuple(shape), tuple(shape_fourier), (6, 6))
        self.shape = shape
        self.samples = samples

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 2-D image.

        Parameters
        ----------
        img: np.ndarray((m, n))
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray((m*n))
            masked Fourier transform of the input image.
        """

        return self.plan.forward(img)

    def adj_op(self, x):
        """ This method calculates the adjoint non-cartesian Fourier
        transform of a 2-D array.

        Parameters
        ----------
        x: np.ndarray((m'*n'))
            masked non-cartesian Fourier transform 2D data.

        Returns
        -------
        img: np.ndarray((m, n))
            adjoint 2D discrete Fourier transform of the input coefficients.
        """
        return self.plan.adjoint(x)
