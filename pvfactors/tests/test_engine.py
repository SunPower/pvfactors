from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
import numpy as np
import datetime as dt


def test_pvengine_float_inputs_iso(params):
    """Test that PV engine works for float inputs"""

    irradiance_model = IsotropicOrdered()
    eng = PVEngine(params, irradiance_model=irradiance_model)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_pvrow'], DNI)

    # Run timestep
    pvarray, vf_matrix, q0, qinc = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
    print(pvarray.pvrows[0].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[1].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[2].front.get_param_weighted('qinc'))


def test_pvengine_float_inputs_perez(params):
    """Test that PV engine works for float inputs"""

    irradiance_model = HybridPerezOrdered()
    eng = PVEngine(params, irradiance_model=irradiance_model)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_pvrow'], DNI)

    # Run timestep
    pvarray, vf_matrix, q0, qinc = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
    print(pvarray.pvrows[0].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[1].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[2].front.get_param_weighted('qinc'))
