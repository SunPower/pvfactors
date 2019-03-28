import numpy as np
from pvfactors.base import BaseSide


def test_baseside(pvsegments):
    """Test that the basic BaseSide functionalities work"""

    side = BaseSide(pvsegments)

    np.testing.assert_array_equal(side.n_vector, [0, 1])
    assert side.shaded_length == 1.
