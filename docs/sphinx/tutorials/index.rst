.. _tutorials:

========================
User guide and tutorials
========================

This section will cover some tutorials to help the users easily get started with ``pvfactors``. The notebooks used for this section are all located in the `tutorials folder`_ of the Github repository.

.. _`tutorials folder`: https://github.com/SunPower/pvfactors/tree/master/docs/tutorials


Understanding PV array 2D geometries
====================================

Let's start with an example of a PV array 2D geometry plotted with ``pvfactors``.


.. figure:: /tutorials//static/Example_pvarray.png
   :align: center
   :width: 50%

   Fig. 1: Example of PV array 2D geometry in ``pvfactors``

As shown in the figure above, a ``pvfactors`` PV array is made out of a list of PV rows (the tilted blue lines), and a ground (the flat lines at y=0).

- Each PV row has 2 sides: a front and a back side, and each side of a PV row is made out of segments. The segments are fixed sections whose location on the PV row side is always constant throughout the simulations, which allows the users to consistently track and calculate irradiance for given sections of a PV row. But each segment of each side of the PV rows is made out of collections of surfaces that are either shaded or illuminated, and these surfaces cannot be tracked during the simulations because they change depending on the PV row rotation angles and the sun's position. In Fig. 1, the leftmost PV row's front side has 3 segments, while its back side has only 1, and the center PV row's back side has 2 segments, while its front side has only 1, etc. For more details on ``pvfactors`` geometries, please refer to the tutorial sections below.
- The ground can be seen as the segment of a "side", which will be made of collections of shaded surfaces (gray lines) and illuminated surfaces (yellow lines) that will change with the PV row rotation angles and the sun angles. Physically, these shaded surfaces represent the shadows of the PV rows that are cast on the ground. The ground will also keep track of "cut points", which are also defined by the PV rows (1 per PV row), and which indicate the extent of the ground that a PV row front side and back side can see.


PV array parameters
===================

In ``pvfactors``, a PV array has a number of fixed parameters that do not change with rotation and solar angles, and which can be passed as a dictionary with specific fields. Below is a sample of a PV array parameters dictionary, which was used to create the 2D geometry shown in Fig. 1.

.. code-block:: python

   pvarray_parameters = {
       'n_pvrows': 3,                            # number of pv rows
       'pvrow_height': 2.5,                      # height of pv rows (measured at center / torque tube)
       'pvrow_width': 2,                         # width of pvrows
       'axis_azimuth': 0.,                       # azimuth angle of rotation axis
       'gcr': 0.4,                               # ground coverage ratio
       'cut': {0: {'front': 3}, 1: {'back': 2}}  # discretization scheme of the pv rows
   }


The :ref:`getting_started_ref` section below shows how such a dictionary can be used to create a PV array in ``pvfactors``. Here is a description of what each parameter means:


- ``n_pvrows``: is the number of PV rows that the PV array will contain. In Fig. 1, we have 3 PV rows.
- ``pvrow_height``: the PV row height (in meters) is the height of the PV row measured from the ground to the PV row center. In Fig. 1, the height of the PV rows is 2.5 m.
- ``pvrow_width``: the PV row width (in meters) is the cross-section width of the entire PV row. In Fig. 1, it's the entire length of the blue lines, so 2 m in the example.
- ``axis_azimuth``: the PV array axis azimuth (in degrees) is the direction of the rotation axis of the PV rows (physically, it could be seen as the torque tube direction for single-axis trackers). In the 2D plane of the PV array geometry (as shown in Fig. 1), it is always the vector normal to that 2D plane and with the direction going into the 2D plane. So positive rotation angles will lead to PV rows tilted to the left, and negative rotation angles will lead to PV rows tilted to the right. The azimuth convention used in ``pvfactors`` is that 0 deg is North, 90 deg is East, etc.
- ``gcr``: it is the ground coverage ratio of the PV array. It is calculated as being equal to the ratio of the PV row width by the distance seperating the PV row centers.
- ``cut``: this optional parameter is used to discretize the PV row sides into equal-length segments. For instance here, the front side of the leftmost PV row (always with index 0) will have 3 segments, and the back side of the center PV row (with index 1) will have 2 segments.

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
