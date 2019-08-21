.. class_organization

.. currentmodule:: pvfactors.geometry

geometry
--------

The geometry sub-package of pvfactors implements multiple classes that make the construction of a 2D geometry for a PV array intuitive and scalable. It is meant to be decoupled from irradiance and view factor calculations so that it can be used independently for other purposes, like visualization for instance. The following schematics summarizes the organization of the classes in this sub-package.

.. figure:: static/class_organization.png
   :align: center
   :width: 80%

base
^^^^

.. automodule:: base
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class_no_base.rst

   base.BaseSurface
   base.PVSurface
   base.ShadeCollection
   base.PVSegment
   base.BaseSide
   base.BasePVArray


pvrow
^^^^^

.. automodule:: pvrow
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class_no_base.rst

   pvrow.PVRowSide
   pvrow.PVRow


pvground
^^^^^^^^

.. automodule:: pvground
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class_no_base.rst

   pvground.PVGround


pvarray
^^^^^^^

.. automodule:: pvarray
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class_no_base.rst

   pvarray.OrderedPVArray


timeseries
^^^^^^^^^^

.. automodule:: timeseries
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class_no_base.rst

   timeseries.TsPVRow
   timeseries.TsGround
   timeseries.TsSide
   timeseries.TsDualSegment
   timeseries.TsSurface
   timeseries.TsLineCoords
   timeseries.TsPointCoords
