.. _class_details:

API Reference
=============

pvfactors.pvarray
-----------------

.. automodule:: pvfactors.pvarray
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: pvfactors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvarray.ArrayBase
   pvarray.Array

Methods
^^^^^^^

.. currentmodule:: pvfactors.pvarray

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Array.update_view_factors
   Array.update_irradiance_terms_perez
   Array.apply_horizon_band_shading
   Array.apply_front_circumsolar_horizon_shading
   Array.update_reflectivity_matrix
   Array.calculate_radiosities_perez
   Array.calculate_sky_and_reflection_components
   Array.create_pvrows_array
   Array.create_pvrow_shadows
   Array.create_ill_ground
   Array.find_edge_points
   Array.create_remaining_illum_ground
   Array.calculate_interrow_direct_shading
   Array.create_surface_registry
   Array.discretize_surfaces
   Array.create_view_matrix


pvfactors.pvcore
----------------

.. automodule:: pvfactors.pvcore
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: pvfactors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvcore.LinePVArray

Functions
^^^^^^^^^

.. currentmodule:: pvfactors.pvcore

.. autosummary::
   :toctree: generated/
   :nosignatures:

   calculate_circumsolar_shading
   integral_default_gaussian
   gaussian_shading
   gaussian
   uniform_circumsolar_disk_shading
   calculate_horizon_band_shading
   find_edge_point


pvfactors.pvgeometry
--------------------

.. automodule:: pvfactors.pvgeometry
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: pvfactors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvgeometry.PVGeometry

Methods and properties
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pvfactors.pvgeometry

Methods

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PVGeometry._series_op
   PVGeometry._geo_unary_op
   PVGeometry._series_unary_op
   PVGeometry.contains
   PVGeometry.distance
   PVGeometry.add
   PVGeometry.split_ground_geometry_from_edge_points
   PVGeometry.break_and_add_entries
   PVGeometry.cut_linestring
   PVGeometry.split_pvrow_geometry
   PVGeometry.cut_pvrow_geometry


Properties

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PVGeometry.area
   PVGeometry.geom_type
   PVGeometry.length
   PVGeometry.boundary
   PVGeometry.centroid
   PVGeometry.bounds


pvfactors.pvrow
---------------

.. automodule:: pvfactors.pvrow
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: pvfactors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvrow.PVRowBase
   pvrow.PVRowLine

Methods and properties
^^^^^^^^^^^^^^^^^^^^^^

Methods

.. currentmodule:: pvfactors.pvrow

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PVRowLine.create_lines
   PVRowLine.get_shadow_bounds
   PVRowLine.calculate_cut_points

Properties

.. currentmodule:: pvfactors.pvrow

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PVRowLine.facing


pvfactors.timeseries
--------------------

.. automodule:: pvfactors.timeseries
   :no-members:
   :no-inherited-members:

Functions
^^^^^^^^^

.. currentmodule:: pvfactors.timeseries

.. autosummary::
   :toctree: generated/
   :nosignatures:

   array_timeseries_calculate
   perez_diffuse_luminance
   calculate_custom_perez_transposition
   calculate_radiosities_serially_perez
   calculate_radiosities_parallel_perez
   get_average_pvrow_outputs
   get_pvrow_segment_outputs
   breakup_df_inputs
   add_df_registries_nans


pvfactors.view_factors
----------------------

.. automodule:: pvfactors.view_factors
   :no-members:
   :no-inherited-members:

Classes
^^^^^^^

.. currentmodule:: pvfactors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   view_factors.ViewFactorCalculator


Methods
^^^^^^^

.. currentmodule:: pvfactors.view_factors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ViewFactorCalculator.calculate_view_factors
   ViewFactorCalculator.vf_hottel_pvrowline_back
   ViewFactorCalculator.vf_hottel_pvrowline_front
   ViewFactorCalculator.vf_trk_to_trk
   ViewFactorCalculator.length_string


Functions
^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   vf_parallel_planes


pvfactors.plot
--------------

.. automodule:: pvfactors.plot
   :no-members:
   :no-inherited-members:

Functions
^^^^^^^^^

.. currentmodule:: pvfactors.plot

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_array_from_registry
   plot_pvarray
   plot_coords
   plot_bounds
   plot_line
