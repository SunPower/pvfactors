from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
import numpy as np
import datetime as dt


def test_pvengine_float_inputs(params):
    """Test that PV engine works for float inputs"""
    eng = PVEngine(params)

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
    pvarray, vf_matrix = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
