from pvfactors.geometry.utils import projection, difference, contains
from pvfactors.geometry.base import BaseSide
from shapely.geometry import Point, LineString, MultiLineString


def test_projection():
    """Make sure that projection method is working as expected"""
    pt_1 = Point(1, 1)
    vector = [0, 1]
    line_1 = LineString([(0, 0), (2, 0)])

    # projection is center
    expected_inter = Point(1, 0)
    inter = projection(pt_1, vector, line_1)
    assert expected_inter == inter

    # should be empty: way outside
    line_2 = LineString([(-1, 0), (0, 0)])
    inter = projection(pt_1, vector, line_2)
    assert inter.is_empty

    # Should be 2nd boundary
    line_3 = LineString([(0, 0), (1, 0)])
    inter = projection(pt_1, vector, line_3)
    assert inter == line_3.boundary[1]

    # Should be 1st boundary
    pt_2 = Point(0, 1)
    inter = projection(pt_2, vector, line_3)
    assert inter == line_3.boundary[0]

    # Should be 1st boundary: very close
    pt_3 = Point(0 + 1e-9, 1)
    inter = projection(pt_3, vector, line_3)
    assert inter == line_3.boundary[0]

    # Should be empty: very close
    pt_4 = Point(0 - 1e-9, 1)
    inter = projection(pt_4, vector, line_3)
    assert inter.is_empty


def test_difference():
    """Testing own implementation of geometry difference operator"""

    # Simple cases
    u = LineString([(0, 0), (2, 0)])

    v = LineString([(1, 0), (3, 0)])
    diff = difference(u, v)
    assert diff == LineString([(0, 0), (1, 0)])

    v = LineString([(3, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(0, 0), (1, 0)])

    v = LineString([(-1, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    v = LineString([(1, 0), (-1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    v = LineString([(0.5, 0), (1.5, 0)])
    diff = difference(u, v)
    assert diff == MultiLineString([((0, 0), (0.5, 0)), ((1.5, 0), (2, 0))])

    v = LineString([(1.5, 0), (0.5, 0)])
    diff = difference(u, v)
    assert diff == MultiLineString([((0, 0), (0.5, 0)), ((1.5, 0), (2, 0))])

    v = LineString([(1, 0), (1, 1)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(1, 1), (1, 0)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(1, 1), (1, 2)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(0, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    # Case with potentially float error
    u = LineString([(0, 0), (3, 2)])
    v = LineString([(0, 0), u.interpolate(0.5, normalized=True)])
    diff = difference(u, v)
    assert diff.length == u.length / 2.

    # Case were should return empty geoemtry
    diff = difference(u, u)
    assert isinstance(diff, LineString)
    assert diff.is_empty

    # Special case that caused crash
    u = LineString([(1, 0), (0, 0)])
    v = LineString([(0, 0), (2, 0)])
    diff = difference(u, v)
    assert diff.is_empty

    # Special case that caused crash
    u = LineString([(1, 0), (0, 0)])
    v = LineString([(-2, 0), (1, 0)])
    diff = difference(u, v)
    assert diff.is_empty


def test_contains_on_side():
    """Check that ``contains`` function works on a BaseSide instance"""
    coords = [(0, 0), (2, 0)]
    side = BaseSide.from_linestring_coords(coords)
    point = Point(1, 0)
    assert contains(side, point)
