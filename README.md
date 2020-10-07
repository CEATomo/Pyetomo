# Pyetomo

Pyetomo is an open-source Python library for compressed sensing tomographic reconstruction using gradient-based and wavelet-based regularizations. It was developped in the context of an interdisciplinary project involving CEA-Leti (Martin Jacob, Jyh-Miin Lin, Zineb Saghi), CEA-Neurospin UNATI (Loubna El Gueddari, Philippe Ciuciu) and CEA-CosmoStat (Samuel Farrens, Jean-Luc Starck). We acknowledge the financial support of the Cross-Disciplinary Program on Numerical Simulation of CEA, the French Alternative Energies and Atomic Energy Commission.

The current version of Pyetomo contains 2D and 3D implementations of:

- Total variation (TV)
- Higher order TV (HOTV)
- Wavelets from PyWavelets library.

PyNUFFT package (https://github.com/jyhmiinlin/pynufft) is used for the Non-Uniform Fast Fourier Transform (NUFFT) operator and its adjoint.

ModOpt package (https://github.com/cea-cosmic/ModOpt), originally developped for MRI and astrophysics applications (http://cosmic.cosmostat.org), gives access to cutting edge optimisation algorithms (e.g.: FISTA, Condat-Vu) and proximity operators.

Pyetomo is intended to be a plugin of PySAP (Python Sparse data Analysis Package). For more information, see: https://github.com/CEA-COSMIC/pysap, and http://cosmic.cosmostat.org/wp-content/uploads/2020/07/Farrens_ASCOM2020.pdf


# Dependencies

During installation, Pyetomo will automatically install the following packages :
 - scipy
 - numpy
 - matplotlib
 - scikit-image
 - pywavelets
 - modopt
 - pynufft


# Installation

To use the package, clone it or download it locally with the following command:

$ git clone https://github.com/CEATomo/Pyetomo.git

Then run :

$ python setup.py install
