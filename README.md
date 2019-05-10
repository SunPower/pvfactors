pvfactors: irradiance modeling made simple
==========================================

<img src="https://raw.githubusercontent.com/SunPower/pvfactors/master/docs/sphinx/_static/logo.png" style="width: 60%;">

[![CircleCI](https://circleci.com/gh/SunPower/pvfactors.svg?style=shield)](https://circleci.com/gh/SunPower/pvfactors)
[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/SunPower/pvfactors/blob/master/LICENSE)

pvfactors is a tool used by PV professionals to calculate the
irradiance incident on surfaces of a photovoltaic array. It relies on the use of
2D geometries and view factors integrated mathematically into a linear system of
equations to account for reflections between all of the surfaces.

pvfactors was originally ported from the SunPower developed 'vf_model' package, which was introduced at the IEEE PV Specialist Conference 44 2017 (see [1] and [link](https://pdfs.semanticscholar.org/ebb2/35e3c3796b158e1a3c45b40954e60d876ea9.pdf) to paper).


Documentation
-------------

The documentation can be found [here](https://sunpower.github.io/pvfactors).
It includes a lot of [tutorials](https://sunpower.github.io/pvfactors/tutorials/index.html) that describe many ways of using pvfactors.


Quick Start
-----------

Given some timeseries inputs:


```python
# Import external libraries
from datetime import datetime
import pandas as pd

# Create input data
df_inputs = pd.DataFrame(
    {'solar_zenith': [20., 50.],
     'solar_azimuth': [110., 250.],
     'surface_tilt': [10., 20.],
     'surface_azimuth': [90., 270.],
     'dni': [1000., 300.],
     'dhi': [50., 500.],
     'albedo': [0.2, 0.2]},
    index=[datetime(2017, 8, 31, 11), datetime(2017, 8, 31, 15)]
)
df_inputs
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>albedo</th>
      <th>dhi</th>
      <th>dni</th>
      <th>solar_azimuth</th>
      <th>solar_zenith</th>
      <th>surface_azimuth</th>
      <th>surface_tilt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-08-31 11:00:00</th>
      <td>0.2</td>
      <td>50.0</td>
      <td>1000.0</td>
      <td>110.0</td>
      <td>20.0</td>
      <td>90.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2017-08-31 15:00:00</th>
      <td>0.2</td>
      <td>500.0</td>
      <td>300.0</td>
      <td>250.0</td>
      <td>50.0</td>
      <td>270.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



And some PV array parameters


```python
pvarray_parameters = {
    'n_pvrows': 3,            # number of pv rows
    'pvrow_height': 1,        # height of pvrows (measured at center / torque tube)
    'pvrow_width': 1,         # width of pvrows
    'axis_azimuth': 0.,       # azimuth angle of rotation axis
    'gcr': 0.4,               # ground coverage ratio
    'rho_front_pvrow': 0.01,  # pv row front surface reflectivity
    'rho_back_pvrow': 0.03    # pv row back surface reflectivity
}
```

The users can run a timeseries simulation using the ``PVEngine``


```python
from pvfactors.engine import PVEngine

# Create engine
engine = PVEngine(pvarray_parameters)
# Fit engine to data
engine.fit(df_inputs.index, df_inputs.dni, df_inputs.dhi,
           df_inputs.solar_zenith, df_inputs.solar_azimuth,
           df_inputs.surface_tilt, df_inputs.surface_azimuth,
           df_inputs.albedo)
```

For a single timestamp


```python
# Import external libraries
import matplotlib.pyplot as plt

# Get pvarray at given timestamp
pvarray = engine.run_timestep(idx=1)

# Plot pvarray shapely geometries
f, ax = plt.subplots(figsize=(10, 3))
pvarray.plot(ax)
plt.show()
```

<img src="https://raw.githubusercontent.com/SunPower/pvfactors/master/docs/sphinx/_static/pvarray.png">


The user can inspect the results very easily thanks to the simple geometry API


```python
print("Incident irradiance on front surface of middle pv row: {} W/m2"
      .format(pvarray.pvrows[1].front.get_param_weighted('qinc')))
print("Reflected irradiance on back surface of left pv row: {} W/m2"
      .format(pvarray.pvrows[0].back.get_param_weighted('reflection')))
print("Isotropic irradiance on back surface of right pv row: {} W/m2"
      .format(pvarray.pvrows[2].back.get_param_weighted('isotropic')))
```

    Incident irradiance on front surface of middle pv row: 811.7 W/m2
    Reflected irradiance on back surface of left pv row: 90.2 W/m2
    Isotropic irradiance on back surface of right pv row: 9.3 W/m2


The users can also run simulations for all timestamps, and obtain a "report" that will look like whatever the users want, and which will rely on the simple geometry API shown above.
Here is an example:


```python
# Create a function that will build a report
from pvfactors.report import example_fn_build_report

# Run full simulation
report = engine.run_all_timesteps(fn_build_report=example_fn_build_report)

# Print results (report is defined by report function passed by user)
df_report = pd.DataFrame(report, index=df_inputs.index)
df_report
```

    100%|██████████| 2/2 [00:00<00:00, 26.58it/s]




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>qinc_back</th>
      <th>iso_back</th>
      <th>qinc_front</th>
      <th>iso_front</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-08-31 11:00:00</th>
      <td>106.627832</td>
      <td>0.115792</td>
      <td>1034.967753</td>
      <td>20.848345</td>
    </tr>
    <tr>
      <th>2017-08-31 15:00:00</th>
      <td>85.535537</td>
      <td>6.304878</td>
      <td>811.659036</td>
      <td>276.181750</td>
    </tr>
  </tbody>
</table>
</div>



Installation
------------

pvfactors is currently compatible and tested with Python versions 2.7 and 3.6, and is available in [PyPI](https://pypi.org/project/pvfactors/).

The easiest way to install pvfactors is to use [pip](https://pip.pypa.io/en/stable/) as follows:

    $ pip install pvfactors

The package wheel files are also available in the [release section](https://github.com/SunPower/pvfactors/releases) of the Github repository.


Requirements
------------

Requirements are included in the ``requirements.txt`` file of the package. Here is
a list of important dependencies:
* [shapely](https://pypi.python.org/pypi/Shapely)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)
* [pandas](https://pypi.python.org/pypi/pandas)
* [pvlib-python](https://pypi.python.org/pypi/pvlib)


Citing pvfactors
----------------

We appreciate your use of pvfactors.
If you use pvfactors in a published work, we kindly ask that you cite:

> Anoma, M., Jacob, D., Bourne, B.C., Scholl, J.A., Riley, D.M. and Hansen, C.W., 2017. View Factor Model and Validation for Bifacial PV and Diffuse Shade on Single-Axis Trackers. In 44th IEEE Photovoltaic Specialist Conference.


Contributing
------------

Contributions are needed in order to improve pvfactors.
If you wish to contribute, you can start by forking and cloning the repository, and then installing pvfactors using [pip](https://pip.pypa.io/en/stable/) in the root folder of the package:

    $ pip install .


To install the package in editable mode, you can use:

    $ pip install -e .


References
----------

[1] Anoma, M., Jacob, D., Bourne, B. C., Scholl, J. A., Riley, D. M., & Hansen, C. W. (2017).
View Factor Model and Validation for Bifacial PV and Diffuse Shade on Single-Axis Trackers. In 44th IEEE Photovoltaic Specialist Conference.
