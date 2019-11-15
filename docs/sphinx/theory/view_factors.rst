.. _view_factors_theory:

View Factors
============

Theory
------

The view factors, also called configuration factors, come from the definition of the directional spectral radiative power of a differential area.

.. figure:: /theory/configuration_factors_pictures/differential_areas.png
   :width: 20%
   :align: center

| Let's take the example of black body surfaces, and then extrapolate the results to more general ones.
| For a black body differential area :math:`dA_1`, we can write that the radiative power emitted to another black body differential area :math:`dA_2` is:

.. math::
	d^2 Q_{{\lambda}, d1-d2}{\;}d{\lambda} = i_{{\lambda},b,1}{\;}d{\lambda}{\;}d{\omega}_1{\;}dA_{1, projected}

| where:
| * :math:`d^2 Q_{{\lambda}, d1-d2}` is the spectral radiative power from :math:`dA_1` to :math:`dA_2`
| * :math:`i_{{\lambda},b,1}` is the blackbody spectral intensity from :math:`dA_1`
| * :math:`{\lambda}` is the wavelength
| * :math:`d{\omega}_1` is the solid angle from :math:`dA_1` to :math:`dA_2`
| * :math:`dA_{1, projected}` is the projected area :math:`dA_1` onto the ``S`` direction

We can then integrate over :math:`{\lambda}` for a black body and rearrange:

.. math::
	d^2 Q_{d1-d2}{\;} = {\frac{{\sigma}{\;}T_1^4}{{\pi}}}{\;}d{\omega}_1{\;}cos{\theta}_1{\;}dA_1

Then:

.. math::
	d^2 Q_{d1-d2}{\;} = {\frac{{\sigma}{\;}T_1^4}{{\pi}}}{\;}{\frac{cos{\theta}_2{\;}dA_2}{S^2}}{\;}cos{\theta}_1{\;}dA_1

And finally:

.. math::
	{\frac{d^2 Q_{d1-d2}}{dA_1}} = {\sigma}{\;}T_1^4 {\frac{cos{\theta}_2{\;}cos{\theta}_1}{{\pi}{\;}S^2}}{\;}dA_2

The view factor from the differential area :math:`dA_1` to the differential area :math:`dA_2` is then defined as:

.. math::
	d^2F_{d1-d2}{\;} = {\frac{cos{\theta}_2{\;}cos{\theta}_1}{{\pi}{\;}S^2}}{\;}dA_2

And for two finite areas :math:`A_1` and :math:`A_2`:

.. math::
	F_{1-2}{\;} = {\frac{1}{A_1}}\int_{A_1}\int_{A_2}d^2F_{d1-d2}{\;}dA_1{\;} = {\frac{1}{A_1}}\int_{A_1}\int_{A_2}{\frac{cos{\theta}_2{\;}cos{\theta}_1}{{\pi}{\;}S^2}}{\;}dA_2{\;}dA_1

We can also note that by reciprocity:

.. math::
	A_1{\;}F_{1-2} = A_2{\;}F_{2-1}

| This approach also holds for diffuse surfaces, whose optical properties don't depend on the direction of the rays.
| We can understand the view factor from a surface :math:`A_1` to a surface :math:`A_2` as the fraction of the hemisphere around :math:`A_1` that is occupied by :math:`A_2`.


Application
-----------

| We will be using configuration factors in the case of 2D geometries, which simplifies the calculations. The 2D assumption is made because the tracker rows considered will be fairly long, and the edge effects will therefore have less impact.
| Also, instead of doing the numerical integration of the double integral representing the view factor, we will systematically try to use analytical solutions of those integrals from tables.
| Here are links describing some view factors relevant to PV array geometries.

* View factor of a wedge: http://www.thermalradiation.net/sectionc/C-5.html
* View factor of parallel planes: http://www.thermalradiation.net/sectionc/C-2a.htm
* View factor of angled planes: http://www.thermalradiation.net/sectionc/C-5a.html
* The Hottel method is also widely used in the model


Adding non-diffuse reflection losses
------------------------------------

For the derivation shown above, we assumed that the surfaces were diffuse. But as shown in [#bifacialvf_paper]_, it is possible to add an approximation of non-diffuse effects by calculating absorption losses that are function of the angle-of-incidence (AOI) of the light.

If we're interested in calculating the **absorbed** irradiance coming from an infinite strip to an infinitesimal surface, we can calculate a view factor derated by AOI losses by starting with the formula derived in http://www.thermalradiation.net/sectionb/B-71.html.


.. figure:: /theory/static/AOI_strips.png
   :align: center
   :width: 40%

   Fig. 1: Schematics illustrating view factor formula from dA1 to infinite strips


The view factor from the infinitesimal surface :math:`dA_1` to the infinite strip :math:`A_{2,1}` is equal to:

.. math::
	dF_{dA_{1}-A_{2,1}}{\;} = {\frac{1}{2}}{\;}({cos{\theta}_2{\;}-{\;}cos{\theta}_1})

For this small view of the strip, we can assume that a given AOI modifier function (:math:`f(AOI)`), which represents reflection losses, is constant. Such that:

.. math::
	dF_{dA_{1}-A_{2,1},AOI}{\;} = {\frac{1}{2}}{\;}f(AOI){\;}({cos{\theta}_2{\;}-{\;}cos{\theta}_1})

We can then calculate the view factor derated by AOI losses from the infinitesimal surface :math:`dA_{1}` to the whole surface :math:`A_{2}` by summing up the values for all the small strips constituting that surface. Such that:

.. math::
	dF_{dA_{1}-A_{2},AOI}{\;} = {\sum}_{j=1}^{3}{\;}dF_{dA_{1}-A_{2,j},AOI}

.. note::
   Since this formula was derived for "infinitesimal" surfaces, in practice we can cut up the PV row sides into "small" segments to make this approximation more valid.


.. [#bifacialvf_paper] Marion, B., MacAlpine, S., Deline, C., Asgharzadeh, A., Toor, F., Riley, D., Stein, J. and Hansen, C., 2017, June. A practical irradiance model for bifacial PV modules. In 2017 IEEE 44th Photovoltaic Specialist Conference (PVSC) (pp. 1537-1542). IEEE.
