import pytest
import numpy as np
from pvfactors import PVFactorsError
from pvfactors.base import BaseSide, ShadeCollection
from pvfactors.pvsurface import PVSurface


def test_baseside(pvsegments):
    """Test that the basic BaseSide functionalities work"""

    side = BaseSide(pvsegments)

    np.testing.assert_array_equal(side.n_vector, [0, 1])
    assert side.shaded_length == 1.


def test_shade_collection():
    """Check that implementation of shade collection works"""
    surf_illum_1 = PVSurface([(0, 0), (1, 0)], shaded=False)
    surf_illum_2 = PVSurface([(0, 0), (0, 1)], shaded=False)
    surf_illum_3 = PVSurface([(1, 0), (2, 0)], shaded=True)

    col = ShadeCollection([surf_illum_1, surf_illum_2])

    assert not col.shaded
    assert col.length == 2

    with pytest.raises(PVFactorsError) as err:
        ShadeCollection([surf_illum_1, surf_illum_3])

    assert str(err.value) \
        == 'All elements should have same shading'
