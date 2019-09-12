.. _irradiance_models_theory:

Irradiance models
=================

As shown in the :ref:`Full mode theory <full_mode_theory>` and :ref:`Fast mode theory <fast_mode_theory>` sections, we always need to calculate a sky term for the different surfaces of the PV array.

The sky term is the sum of all the irradiance components (for each surface) that are not directly related to the view factors or to the reflection process, but which still contribute to the incident irradiance on the surfaces. For instance, the direct component of the light incident on the front surface of a PV row is not directly dependent on the view factors, but we still need to account for it in the mathematical model, so this component will go into the sky term.

A lot of different assumptions can be made, which will lead to more or less accurate results. But ``pvfactors`` was designed to make the implementation of these assumptions modular: all of these assumptions can be implemented inside a single Python class which can be used by the other parts of the model. This was done to make it easy for users to create their own irradiance modeling assumptions (inside a new class), and to then plug it into the ``pvfactors`` view-factor mathematical and geometry models.

``pvfactors`` currently provides two irradiance models that can be used interchangeably in the calculations, and which are described in more details in the :ref:`irradiance developer API <irradiance_classes>`.

- the isotropic model assumes that all of the diffuse light from the sky dome is isotropic.
- the (hybrid) perez model follows [#perez_paper]_ and assumes that the diffuse light can be broken down into circumsolar, isotropic, and horizon components (see Fig. 1 below). Validation work shows that this model is more accurate for calculating back-side irradiance with ``pvfactors``.

.. figure:: /theory/static/Irradiance_components.PNG
   :align: center
   :width: 40%

   Fig. 1: Schematic showing direct and diffuse irradiance components on a PV system and according to the Perez diffuse light model [#perez_paper]_



.. rubric:: Footnotes

.. [#perez_paper] Perez, R., Seals, R., Ineichen, P., Stewart, R. and Menicucci, D., 1987. A new simplified version of the Perez diffuse irradiance model for tilted surfaces. Solar energy, 39(3), pp.221-231.
