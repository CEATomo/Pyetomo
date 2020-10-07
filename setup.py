#! /usr/bin/env python
##########################################################################
# pySAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import os
from setuptools import setup, find_packages


# Global parameters
PKGDATA = {
    "etomo-plugin": [
        "pyetomo"
    ]
}
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]
AUTHOR = """
Martin Jacob <martin.jacob@cea.fr>
Zineb Saghi <zineb.saghi@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""

# Write setup
setup(
    name="python-pyetomo",
    description="Python library for compressed sensing tomographic reconstruction.",
    long_description="Open-source Python library for compressed sensing tomographic reconstruction, using gradient-based and wavelet-based regularizations.",
    license="CeCILL-B",
    classifiers="CLASSIFIERS",
    author=AUTHOR,
    author_email="martin.jacob@cea.fr; zineb.saghi@cea.fr; philippe.ciuciu@cea.fr",
    version="2.0",
    url="https://github.com/CEATOmo/Pyetomo",
    packages=find_packages(),
    platforms="OS Independent",
    package_data=PKGDATA,
    install_requires=['scipy','numpy','matplotlib','scikit-image','pywavelets','modopt','pynufft'],
)
