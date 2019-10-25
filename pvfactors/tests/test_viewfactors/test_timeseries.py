from pvfactors.viewfactors.calculator import VFCalculator


def test_ts_vf_matrix(ordered_pvarray):
    """Test that timeseries vf matrix is calculated correctly"""
    vfcalculator = VFCalculator()
    vf_matrix = vfcalculator.build_ts_vf_matrix(ordered_pvarray)

    # Check that correct size
    assert vf_matrix.shape == (40, 40, 1)
