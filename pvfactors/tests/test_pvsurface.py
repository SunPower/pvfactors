from pvfactors.pvsurface import PVSurface, PVSegment
from pvfactors.pvrow import PVRowSide


def test_pvsurfacefluid_setter():
    """Test the fluid pv surface collection updates correctly"""
    surf = PVSegment()
    assert surf.length == 0
    new_surf_rigid = PVSurface([(0, 0), (1, 0)], shaded=False)
    surf.illum_surface = new_surf_rigid
    assert surf.length == 1
    new_surf_rigid = PVSurface([(1, 0), (2, 0)], shaded=True)
    surf.shaded_surface = new_surf_rigid
    assert surf.length == 2


def test_pvsurfacefluid_deleter():
    """Test that elements of the fluid pv surface collection get deleted
    correctly"""
    surf = PVSegment(PVSurface([(0, 0), (1, 0)], shaded=False),
                     PVSurface([(1, 0), (2, 0)], shaded=True))
    assert surf.length == 2
    del surf.shaded_surface
    assert surf.length == 1
    del surf.illum_surface
    assert surf.length == 0


def test_shaded_length():
    """Test that calculation of shaded length is correct"""
    surf_1 = PVSegment(
        illum_surface=PVSurface([(0, 0), (1, 0)], shaded=False))
    assert surf_1.shaded_length == 0
    surf_2 = PVSegment(
        shaded_surface=PVSurface([(1, 0), (2, 0)], shaded=True))
    assert surf_2.shaded_length == 1

    side = PVRowSide([surf_1, surf_2])
    assert side.shaded_length == 1
