# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains classes for defining gradients operators.
"""
import numpy as np

from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class GradAnalysis(GradBasic, PowerMethod):
    """
    Gradient operator for the data attachment term in the reconstruction
    process in analysis formulation
    """

    def __init__(self, data, fourier_op):
        GradBasic.__init__(self, data, fourier_op.op, fourier_op.adj_op)
        self.fourier_op = fourier_op
        PowerMethod.__init__(self, self.trans_op_op, self.fourier_op.shape,
                             data_type=complex, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)


class GradSynthesis(GradBasic, PowerMethod):
    """
    Gradient operator for the data attachment term in the reconstruction
    process in synthesis formulation
    """

    def __init__(self, data, linear_op, fourier_op):
        GradBasic.__init__(self, data, self._op_method, self._trans_op_method)
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(fourier_op.shape).astype(complex))
        PowerMethod.__init__(self, self.trans_op_op, coef.shape,
                             data_type=complex, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)

    def _op_method(self, data, *args, **kwargs):
        # pylint: disable=unused-argument
        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data, *args, **kwargs):
        # pylint: disable=unused-argument
        return self.linear_op.op(self.fourier_op.adj_op(data))
