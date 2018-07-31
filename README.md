pvfactors (open-source fork of vf_model)
========================================

pvfactors is a tool designed for PV professionals to calculate the
irradiance incident on surfaces of a photovoltaic array. It relies on the use of
2D geometries and view factors integrated mathematically into a linear system of
equations.

This package is the open-source fork of the original 'vf_model' package developed
by SunPower, and which had over 300 commits. The package was used for all the
material presented at IEEE PVSC 44 2017 (see [1]).


Documentation
-------------

The documentation can be found [here](https://sunpower.github.io/pvfactors).


Installation
------------

pvfactors is currently compatible and tested with Python versions 2.7 and 3.6

You can install the package using [pip](https://pip.pypa.io/en/stable/) and the
wheel file from the latest release in the [release section](https://github.com/SunPower/pvfactors/releases) of this Github repository.

You can also fork this repository and install the pvfactors using [pip](https://pip.pypa.io/en/stable/) in the root folder of the package:

    $ pip install .


To install the package in editable mode, you can use:

    $ pip install -e .


Requirements
------------

Requirements are included in the ``setup.py`` file of the package. Here is
a list of important dependencies:
* [shapely](https://pypi.python.org/pypi/Shapely)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)
* [pandas](https://pypi.python.org/pypi/pandas)
* [pvlib-python](https://pypi.python.org/pypi/pvlib)


Quickstart Example
------------------

The following Jupyter notebook is a good way to get started: [notebook](http://sunpower.github.io/pvfactors/developer/pvfactors_demo.html)


References
----------

[1] Marc Abou Anoma, David Jacob, Ben C. Bourne, Jonathan A. Scholl,
Daniel M. Riley, Clifford W. Hansen. “View Factor Model and Validation
for Bifacial PV and Diffuse Shade on Single-Axis Trackers.”
Photovoltaic Specialist Conference (PVSC), 2017 IEEE 44th. IEEE, 2017.
