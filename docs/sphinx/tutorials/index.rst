.. _tutorials:

========================
User guide and tutorials
========================

This section will cover some tutorials to help the users easily get started with ``pvfactors``. The notebooks used for this section are all located in the `tutorials folder`_ of the Github repository.

.. _`tutorials folder`: https://github.com/SunPower/pvfactors/tree/master/docs/tutorials


.. _getting_started_ref:

Getting started: running simulations
====================================

Here is a quick overview on how to get started and run irradiance simulations with ``pvfactors``. The users may find it useful to first read the theory and mathematical formulation for :ref:`full_mode_theory` and :ref:`fast_mode_theory` to understand the difference between the two approaches.


.. toctree::
   :maxdepth: 1

   Getting_started


Details on the "fast mode" simulations
======================================

In "fast mode" (and as described in the :ref:`fast_mode_theory` theory section), ``pvfactors`` assumes that all incident irradiance values for the system are known except for the PV row back surfaces. So since the system to solve is now explicit (no matrix inversion needed), vectorization is used to speed up the calculations.

This mode relies on "timeseries geometries" of the PV array, which are the attributes named ``ts_pvrows`` and ``ts_ground``, and which contain vectors of coordinates for all timestamps and for all geometry elements. Please take a look at the tutorial sections below for more details on the fast mode.

.. toctree::
   :maxdepth: 1

   Run_fast_simulations


Details on the "full mode" simulations
======================================

In "full mode" (and as described in the :ref:`full_mode_theory` theory section), ``pvfactors`` calculates the equilibrium of reflections between all surfaces of the PV array for each timestamp. So the system to solve is implicit (matrix inversion required). This mode relies on timestamp specfic geometries for the PV array using its attributes ``pvrows`` and ``ground``.

.. toctree::
   :maxdepth: 1

   PVArray_introduction
   Create_discretized_pvarray
   Calculate_view_factors
   Run_full_timeseries_simulations
   Run_full_parallel_simulations
