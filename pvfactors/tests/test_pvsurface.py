from pvfactors.pvsurface import PVSurfaceRigid, PVSurfaceFluid


def test_pvsurfacefluid_setter():
    """Test the fluid pv surface collection updates correctly"""
    surf = PVSurfaceFluid()
    assert surf.length == 0
    new_surf_rigid = PVSurfaceRigid([(0, 0), (1, 0)], shaded=False)
    surf.illum_surface = new_surf_rigid
    assert surf.length == 1
    new_surf_rigid = PVSurfaceRigid([(1, 0), (2, 0)], shaded=True)
    surf.shaded_surface = new_surf_rigid
    assert surf.length == 2


def test_pvsurfacefluid_deleter():
    """Test that elements of the fluid pv surface collection get deleted
    correctly"""
    surf = PVSurfaceFluid(PVSurfaceRigid([(0, 0), (1, 0)], shaded=False),
                          PVSurfaceRigid([(1, 0), (2, 0)], shaded=True))
    assert surf.length == 2
    del surf.shaded_surface
    assert surf.length == 1
    del surf.illum_surface
    assert surf.length == 0
