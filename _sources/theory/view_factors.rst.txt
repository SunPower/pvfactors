View Factors
============

Theory
------

The view factors, also called configuration factors, come from the definition of the directional spectral radiative power of a differential area.

.. figure:: /theory/configuration_factors_pictures/differential_areas.png
	:width: 30%

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
