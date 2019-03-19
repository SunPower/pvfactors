.. _problem_formulation:

Problem Formulation
===================

In order to use the view factors as follows, we need to assume that the surfaces considered are diffuse (lambertian). Which means that their optical properties are independent of the angle of the rays (incident, reflected, or emitted). This is an important assumption that prevents the usage of AOI loss factors for instance.

The current version of the view factor model only addresses PV rows that are made out of straight lines (no "dual-tilt" for instance). But the PV array can have any azimuth or tilt angle for the simulations. Below is the 2D representation of such a PV array.


.. figure:: /theory/problem_formulation_pictures/Array_example_clean.png
   :width: 60%
   :align: center

By making some assumptions, it is possible to represent the calculation of irradiance terms on each surface with a linear system. The dimension of this system changes depending on the number of surfaces considered. But we can formulate it for the general case of ``n`` surfaces.

For a surface ``i`` we can write that:

.. math:: q_{o, i} = q_{emitted, i} + q_{reflected, i}

Unit: :math:`W/m^2`.

| * :math:`q_{o, i}` is the radiosity of surface ``i``, and it represents the outgoing radiative flux from it.
| * :math:`q_{emitted, i}` is the emitted radiative flux from that surface. For instance the total emitted radiative flux of a blackbody is known to be :math:`{\sigma}T^4` (with :math:`T` the surface temperature and :math:`{\sigma}` the Stefan–Boltzmann constant).
| * :math:`q_{reflected, i}` is the reflected flux from that surface.

Finding values of interest like back side irradiance can only be done after finding the radiosities :math:`q_{o, i}` of all surfaces ``i``. This can become a very complex system of equations where one would need to solve the energy balance on the considered systems .

| But if we decide to make the assumption that :math:`q_{emitted, i}` is negligible, we can simplify the problem in a way that would enable us to find more easily some approximations of the values of interest. For now, this assumption makes some sense because the temperatures of the PV systems and the surroundings are generally not very high (< 330K). Besides the surfaces are not real black bodies, which means that their total (or broadband) emissions and absorptions will be even lower.
| Under this assumption, we end up with:

.. math:: q_{o, i}{\;}{\approx}{\;}q_{reflected, i}

where:

.. math:: q_{reflected, i} = {\rho_i} * q_{incident, i}

| with:
| * :math:`q_{incident, i}` is the incident radiative flux on surface ``i``.
| * :math:`{\rho_i}` is the total reflectivity of surface ``i``.

We can further develop this expression and involve configuration factors as well as irradiance terms as follows:

.. math:: q_{reflected, i} = {\rho_i} * ({\sum_{j} q_{o, j} * F_{i, j}} + Irr_i)

| where:
| * :math:`{\sum_{j} q_{o, j} * F_{i, j}}` is the contribution of all the surfaces ``j`` surrounding ``i`` to the incident radiative flux onto surface ``i``.
| * :math:`F_{i, j}` is the configuration factor (or view factor) of surface ``i`` to surface ``j``.
| * :math:`Irr_i` is an irradiance term specific to surface ``i`` which contributes to the incident radiative flux  :math:`q_{incident, i}`. For instance, it will be equal to the :math:`DNI_{POA}` for the front side of the modules.

This results into a linear system that can be written as follows:

.. math::

	\mathbf{q_o} = \mathbf{R} . (\mathbf{F} . \mathbf{q_o} + \mathbf{Irr})

	(\mathbf{R}^{-1} - \mathbf{F}).\mathbf{q_o} = \mathbf{Irr}

Or, for a system of ``n`` surfaces:

.. math::

	(\begin{pmatrix}
	{\rho_1}      & 0             & 0      & \cdots   & 0\\
	0             & {\rho_2}      & 0      & \cdots   & 0\\
	\vdots        & \vdots        & \vdots & \ddots   & \vdots\\
	0             & 0             & 0      & \cdots   & {\rho_n}\\
	\end{pmatrix}^{-1} -
	\begin{pmatrix}
	F_{1,1}      & F_{1,2}      & F_{1,3}      & \cdots   & F_{1,n}\\
	F_{2,1}      & F_{2,2}      & F_{2,3}      & \cdots   & F_{2,n}\\
	\vdots       & \vdots       & \vdots       & \ddots   & \vdots\\
	F_{n,1}      & F_{n,2}      & F_{n,3}      & \cdots   & F_{n,n}\\
	\end{pmatrix}).
	\begin{pmatrix}
	q_{o, 1}\\
	q_{o, 2}\\
	\vdots\\
	q_{o, n}\\
	\end{pmatrix}
	=
	\begin{pmatrix}
	Irr_1\\
	Irr_2\\
	\vdots\\
	Irr_n\\
	\end{pmatrix}

After solving this system and finding all of the radiosities, it is very easy to deduce values of interest like back side or front side incident irradiance.
