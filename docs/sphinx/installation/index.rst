.. _installation:


Installation
============

Install with pip
----------------

``pvfactors`` works with both python 2 and 3.

The easiest way to install ``pvfactors`` is using pip_:

.. code-block:: shell

   $ pip install pvfactors

However, installing ``shapely`` from PyPI may not install all the necessary binary dependencies.
If you run into an error like ``OSError: [WinError 126] The specified module could not be found``,
try installing conda from conda-forge with:

.. code-block:: shell

   $ conda install -c conda-forge shapely

Windows users may also be able to resolve the issue by installing wheels from `Christoph Gohlke`_.

.. _Christoph Gohlke: https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely


pvlib implementation
--------------------

A limited implementation of ``pvfactors`` is available in the bifacial module of pvlib-python_: see here_.

.. _pvlib-python: https://pvlib-python.readthedocs.io
.. _here: https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.bifacial.pvfactors_timeseries.html#pvlib.bifacial.pvfactors_timeseries


Contributing
------------

Contributions are needed in order to improve this package.
If you wish to contribute, you can start by forking and cloning the repository, and then installing ``pvfactors`` using pip_ in the root folder of the package:

.. code-block:: shell

   $ pip install .


To install the package in editable mode, you can use:

.. code-block:: shell

   $ pip install -e .


.. _pip: https://pypi.org/project/pip/
