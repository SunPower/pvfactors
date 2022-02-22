.. _tutorials:

=========
Tutorials
=========

This section will cover some tutorials to help the users easily get started with ``pvfactors``. The notebooks used for this section are all located in the `tutorials folder`_ of the Github repository.

.. note::

   The users may find it useful to first read the theory and mathematical formulation for :ref:`full_mode_theory` and :ref:`fast_mode_theory` to better understand the differences between the two approaches.


.. _`tutorials folder`: https://github.com/SunPower/pvfactors/tree/master/docs/tutorials


.. _getting_started_ref:

Getting started: running simulations
====================================

Here is a quick overview on how to get started and run irradiance simulations with ``pvfactors``.


.. toctree::
   :maxdepth: 1

   Getting_started


Details on the "full mode" simulations
======================================

In the "full mode", ``pvfactors`` calculates the equilibrium of reflections between all surfaces of the PV array for each timestamp. So the system to solve is implicit (matrix inversion required).

``pvfactors`` relies on "timeseries geometries" of the PV array, which are the attributes named ``ts_pvrows`` and ``ts_ground`` in :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`, and which contain vectors of coordinates for all timestamps and for all geometry elements. Please take a look at the tutorial sections below for more details on this.

.. toctree::
   :maxdepth: 1

   PVArray_introduction
   Create_discretized_pvarray
   Calculate_view_factors
   Run_full_timeseries_simulations
   Run_full_parallel_simulations
   Account_for_AOI_losses


Details on the "fast mode" simulations
======================================

In the "fast mode", ``pvfactors`` assumes that all incident irradiance values for the system are known except for the PV row back surfaces. So since the system to solve is now explicit (no matrix inversion needed), it runs a little bit faster than the full mode, but it is less accurate.

.. note::
   Some tests show that for 8760 hourly simulations, the run time is less than 1 second for the fast mode vs. less than 2 seconds for the full mode.


.. toctree::
   :maxdepth: 1

   Run_fast_simulations
