.. class_organization

.. currentmodule:: pvfactors.geometry

Geometry API
============

The geometry sub-package of pvfactors implements multiple classes that make the construction of a 2D geometry for a PV array intuitive and scalable. It is meant to be decoupled from irradiance and view factor calculations so that it can be used independently for other purposes, like visualization for instance. The following schematics summarizes the organization of the classes in this sub-package.

.. figure:: static/class_organization.png
   :align: center
   :width: 80%

geometry.base
-------------

.. automodule:: base
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   base.BaseSurface
   base.PVSurface
   base.ShadeCollection
   base.PVSegment
   base.BaseSide


geometry.pvrow
--------------

.. automodule:: pvrow
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvrow.PVRowSide
   pvrow.PVRow


geometry.pvground
-----------------

.. automodule:: pvground
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvground.PVGround


geometry.pvarray
----------------

.. automodule:: pvarray
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pvarray.OrderedPVArray
