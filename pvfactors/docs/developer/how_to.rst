.. _how_to:

How to use the model for single-point calculation
=================================================

| Below is a description of the main steps order to run a calculation for a single (time) point. The following schematic summarizes it. This section should be used as a complement to the following Python Jupyter notebook guide: :doc:`/developer/pvfactors_demo`
| The notebook also describes how to use the model for timeseries calculations (as opposed to single time point).


.. figure:: /developer/class_implementation_pictures/pvfactors_process.png

Instantiate PV array class
--------------------------

| In order to use the model, the first step is to instantiate the ``Array`` class using the desired geometry parameters. Based on the inputs, the algorithm will create a 2D model of the PV array using ``shapely`` objects.
| The class will also automatically calculate the view factors for the specified configuration and rotation angles.

.. autoclass:: pvarray.Array
   :noindex:

Visualize PV array lines
------------------------

| The user can easily visualize the array by plotting the ``shapely`` lines that it's comprised of.
| The ``plot_pvarray`` and ``plot_array_from_registry`` functions located in the ``plot`` module should be used in combination with the ``matplotlib`` package for visualization.

.. autofunction:: plot.plot_pvarray
   :noindex:


Update PV array angles
----------------------

| After creating a PV array object, the user can update the rotation and solar angles using the ``update_view_factors`` method.

.. automethod:: pvarray.Array.update_view_factors
   :noindex:

Calculate irradiance terms
--------------------------

After creating a PV array object, the user can calculate all of the irradiance terms described in the problem formulation.
The ouputs of the calculation will be written into the ``surface_registry``. The approach builds on top of the Perez diffuse light transposition model.

.. automethod:: pvarray.Array.calculate_radiosities_perez
   :noindex:

Surface registry
----------------

In the end, the core information and results about the PV array object will be stored in its ``surface_registry`` attribute of the ``Array`` object.
