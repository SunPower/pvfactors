"""Base classes for geometry subpackage."""

import numpy as np
import pandas as pd
from pvfactors import PVFactorsError
from pvfactors.config import (
    DEFAULT_NORMAL_VEC, COLOR_DIC, DISTANCE_TOLERANCE, PLOT_FONTSIZE,
    ALPHA_TEXT)
from pvfactors.geometry.plot import plot_coords, plot_bounds, plot_line
from pvfactors.geometry.utils import \
    is_collinear, check_collinear, are_2d_vecs_collinear, difference, contains
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry.collection import geos_geometrycollection_from_py
from shapely.ops import linemerge
from pvlib.tools import cosd, sind


def check_uniform_shading(list_elements):
    """Check that all :py:class:`~pvfactors.pvsurface.PVSurface` objects in
    list are collinear"""
    shaded = None
    for element in list_elements:
        if shaded is None:
            shaded = element.shaded
        else:
            is_uniform = shaded == element.shaded
            if not is_uniform:
                msg = "All elements should have same shading"
                raise PVFactorsError(msg)


def coords_from_center_tilt_length(xy_center, tilt, length,
                                   surface_azimuth, axis_azimuth):
    """Calculate ``shapely`` :py:class:`LineString` coordinate from
    center coords, surface tilt angle and length of line.
    The axis azimuth indicates the axis of rotation of the pvrows (if single-
    axis trackers). In the 2D plane, the axis of rotation will be the vector
    normal to that 2D plane and going into the 2D plane (when plotting it).
    The surface azimuth should always be 90 degrees away from the axis azimuth,
    either in the positive or negative direction.
    For instance, a single axis trk with axis azimuth = 0 deg (North), will
    have surface azimuth values equal to 90 deg (East) or 270 deg (West).
    Tilt angles need to always be positive. For the axis azimuth and surface
    azimuth, a rotation angle will be derived. Positive rotation angles will
    indicate pvrows pointing to the left, and negative rotation angles will
    indicate pvrows pointing to the right.
    All of these conventions are necessary to make sure that no matter what
    the tilt and surface angles are, we can still identify correctly
    the same pv rows.
    """
    is_pointing_right = ((surface_azimuth - axis_azimuth) % 360.) > 180.
    if is_pointing_right:
        # system should be rotated to the right: so negative rotation angle
        rotation = tilt
    else:
        # need positive rotation angle for system to point to the left
        rotation = - tilt
    x_center, y_center = xy_center
    radius = length / 2.
    x1 = radius * cosd(rotation + 180.) + x_center
    y1 = radius * sind(rotation + 180.) + y_center
    x2 = radius * cosd(rotation) + x_center
    y2 = radius * sind(rotation) + y_center

    return [(x1, y1), (x2, y2)]


def get_solar_2d_vector(solar_zenith, solar_azimuth, axis_azimuth):
    """Projection of 3d solar vector onto the cross section of the systems:
    which is the 2d plane we are considering: needed to calculate shadows
    Remember that the 2D plane is such that the direction of the torque
    tube vector goes into (and normal to) the 2D plane, such that
    positive rotation angles will have the PV surfaces tilted to the LEFT
    and vice versa"""
    solar_2d_vector = np.array([
        # a drawing really helps understand the following
        sind(solar_zenith) * cosd(solar_azimuth - axis_azimuth - 90.),
        cosd(solar_zenith)])

    return solar_2d_vector


class BaseSurface(LineString):
    """Base surfaces will be extensions of :py:class:`LineString` classes,
    but adding an orientation to it. So two surfaces could use the same
    linestring, but have opposite orientations."""

    def __init__(self, coords, normal_vector=None, index=None):
        """Normal vector can have two directions for a given LineString,
        so the user can provide it in order to be specific,
        otherwise it will be automatically
        calculated, but then the surface won't know if it was supposed to be
        pointing "up" or "down". If the surface is empty, the normal vector
        will take the default value."""

        super(BaseSurface, self).__init__(coords)
        if normal_vector is None:
            self.n_vector = self._calculate_n_vector()
        else:
            self.n_vector = np.array(normal_vector)
        self.index = index

    def _calculate_n_vector(self):
        """Calculate normal vector of the surface if not empty"""
        if not self.is_empty:
            b1, b2 = self.boundary
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            return np.array([-dy, dx])
        else:
            return DEFAULT_NORMAL_VEC

    def plot(self, ax, color=None, with_index=False):
        plot_coords(ax, self)
        plot_bounds(ax, self)
        plot_line(ax, self, color)
        if with_index:
            # Prepare text location
            v = self.n_vector
            v_norm = v / np.linalg.norm(v)
            centroid = self.centroid
            alpha = ALPHA_TEXT
            x = centroid.x + alpha * v_norm[0]
            y = centroid.y + alpha * v_norm[1]
            # Add text
            ax.text(x, y, '{}'.format(self.index),
                    verticalalignment='center', horizontalalignment='center')

    def difference(self, linestring):
        """Calculate remaining surface after removing part belonging from
        provided linestring"""
        return difference(self, linestring)


class PVSurface(BaseSurface):
    """Rigid PV surfaces are extensions of base surfaces, as they add PV
    related properties and methods, like reflectivity, shading, etc. They can
    only represent only 1 ``shapely`` :py:class:`LineString` geometry."""

    def __init__(self, coords=None, normal_vector=None, shaded=False,
                 index=None):

        super(PVSurface, self).__init__(coords, normal_vector, index=index)
        self.shaded = shaded


class ShadeCollection(GeometryCollection):
    """A group :py:class:`~pvfactors.pvsurface.PVSurface`
    objects that have the same shading status. The pv surfaces are not
    necessarily contiguous or collinear."""

    def __init__(self, list_surfaces=[], shaded=None):
        """
        Parameters
        ----------
        shaded : bool, optional
            Will be used only if no surfaces were passed
        """
        check_uniform_shading(list_surfaces)
        self.list_surfaces = list_surfaces
        self.shaded = self._get_shading(shaded)
        self.is_collinear = is_collinear(list_surfaces)
        super(ShadeCollection, self).__init__(list_surfaces)

    def _get_shading(self, shaded):
        """Get the surface shading from the provided list of pv surfaces."""
        if len(self.list_surfaces):
            return self.list_surfaces[0].shaded
        else:
            return shaded

    def plot(self, ax, color=None, with_index=False):
        for surface in self.list_surfaces:
            surface.plot(ax, color=color, with_index=with_index)

    def add_linestring(self, linestring):
        surf = PVSurface(coords=linestring.coords, normal_vector=self.n_vector,
                         shaded=self.shaded)
        self.add_pvsurface(surf)

    def add_pvsurface(self, pvsurface):
        assert pvsurface.shaded == self.shaded
        self.list_surfaces.append(pvsurface)
        self.is_collinear = is_collinear(self.list_surfaces)
        super(ShadeCollection, self).__init__(self.list_surfaces)

    def remove_linestring(self, linestring):
        new_list_surfaces = []
        for surface in self.list_surfaces:
            # Need to use buffer for intersects bc of floating point precision
            # errors in shapely
            if surface.buffer(DISTANCE_TOLERANCE).intersects(linestring):
                difference = surface.difference(linestring)
                # We want to make sure we can iterate on it, as
                # ``difference`` can be a multi-part geometry or not
                if not hasattr(difference, '__iter__'):
                    difference = [difference]
                for new_geom in difference:
                    if not new_geom.is_empty:
                        new_surface = PVSurface(new_geom.coords,
                                                normal_vector=surface.n_vector,
                                                shaded=surface.shaded)
                        new_list_surfaces.append(new_surface)
            else:
                new_list_surfaces.append(surface)

        self.list_surfaces = new_list_surfaces
        # Force update, even if list is empty
        self.update_geom_collection(self.list_surfaces)

    def update_geom_collection(self, list_surfaces):
        """Force update of geometry collection, even if list is empty
        https://github.com/Toblerity/Shapely/blob/master/shapely/geometry/collection.py#L42
        """
        self._geom, self._ndim = geos_geometrycollection_from_py(list_surfaces)

    def merge_surfaces(self):
        """Merge all surfaces in the shade collection, even if they're not
        contiguous, by using bounds, into one contiguous surface"""
        if len(self.list_surfaces) > 1:
            merged_lines = linemerge(self.list_surfaces)
            minx, miny, maxx, maxy = merged_lines.bounds
            new_pvsurf = PVSurface(
                coords=[(minx, miny), (maxx, maxy)],
                shaded=self.shaded,
                normal_vector=self.list_surfaces[0].n_vector)
            self.list_surfaces = [new_pvsurf]
            self.update_geom_collection(self.list_surfaces)

    def cut_at_point(self, point):
        """Cut collection at point if contains it"""
        for idx, surface in enumerate(self.list_surfaces):
            if contains(surface, point):
                # Make sure that not hitting a boundary
                b1, b2 = surface.boundary
                not_hitting_b1 = b1.distance(point) > DISTANCE_TOLERANCE
                not_hitting_b2 = b2.distance(point) > DISTANCE_TOLERANCE
                if not_hitting_b1 and not_hitting_b2:
                    coords_1 = [b1, point]
                    coords_2 = [point, b2]
                    # TODO: not sure what to do about index yet
                    new_surf_1 = PVSurface(coords_1,
                                           normal_vector=surface.n_vector,
                                           shaded=surface.shaded)
                    new_surf_2 = PVSurface(coords_2,
                                           normal_vector=surface.n_vector,
                                           shaded=surface.shaded)
                    # Now update collection
                    self.list_surfaces[idx] = new_surf_1
                    self.list_surfaces.append(new_surf_2)
                    self.update_geom_collection(self.list_surfaces)
                    # No need to continue the loop
                    break

    @property
    def n_vector(self):
        if not self.is_collinear:
            msg = "Cannot request n_vector if all elements not collinear"
            raise PVFactorsError(msg)
        if len(self.list_surfaces):
            return self.list_surfaces[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        """Number of surfaces in collection"""
        return len(self.list_surfaces)

    @property
    def surface_indices(self):
        return [surf.index for surf in self.list_surfaces]

    @classmethod
    def from_linestring_coords(cls, coords, shaded, normal_vector=None):
        surf = PVSurface(coords=coords, normal_vector=normal_vector,
                         shaded=shaded)
        return cls([surf], shaded=shaded)


class PVSegment(GeometryCollection):
    """A PV segment will be a collection of 2 collinear and contiguous
    PV surfaces, a shaded one and an illuminated one. It inherits from
    ``shapely``
    :py:class:`GeometryCollection` so that we can still call basic geometrical
    methods and properties on it, eg call length and get sum of lengths
    here of the two surfaces"""

    def __init__(self, illum_collection=ShadeCollection(shaded=False),
                 shaded_collection=ShadeCollection(shaded=True), index=None):
        assert shaded_collection.shaded, "surface should be shaded"
        assert not illum_collection.shaded, "surface should not be shaded"
        self._check_collinear(illum_collection, shaded_collection)
        self._shaded_collection = shaded_collection
        self._illum_collection = illum_collection
        self.index = index
        self._all_surfaces = None
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    def _check_collinear(self, illum_collection, shaded_collection):
        assert illum_collection.is_collinear
        assert shaded_collection.is_collinear
        # Check that if none or all of the collection is empty, n_vectors are
        # equal
        if (not illum_collection.is_empty) \
           and (not shaded_collection.is_empty):
            n_vec_ill = illum_collection.n_vector
            n_vec_shaded = shaded_collection.n_vector
            assert are_2d_vecs_collinear(n_vec_ill, n_vec_shaded)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        self._shaded_collection.plot(ax, color=color_shaded,
                                     with_index=with_index)
        self._illum_collection.plot(ax, color=color_illum,
                                    with_index=with_index)

    def cast_shadow(self, linestring):
        """Split up segment from a linestring object: will rearrange the
        PV surfaces between the shaded and illuminated collections of the
        segment"""
        # Using a buffer may slow things down, but it's quite crucial
        # in order for shapely to get the intersection accurately see:
        # https://stackoverflow.com/questions/28028910/how-to-deal-with-rounding-errors-in-shapely
        intersection = (self._illum_collection.buffer(DISTANCE_TOLERANCE)
                        .intersection(linestring))
        if not intersection.is_empty:
            # Split up only if interesects the illuminated collection
            # print(intersection)
            self._shaded_collection.add_linestring(intersection)
            # print(self._shaded_collection.length)
            self._illum_collection.remove_linestring(intersection)
            # print(self._illum_collection.length)
            super(PVSegment, self).__init__([self._shaded_collection,
                                             self._illum_collection])

    def cut_at_point(self, point):
        """Cut segment at a given point, only if contained by segment"""
        if contains(self, point):
            if contains(self._illum_collection, point):
                self._illum_collection.cut_at_point(point)
            else:
                self._shaded_collection.cut_at_point(point)

    @property
    def n_vector(self):
        """Since shaded and illum surfaces are supposed to be collinear,
        this should return either surfaces' normal vector. If both empty,
        return None."""
        if not self.illum_collection.is_empty:
            return self.illum_collection.n_vector
        elif not self.shaded_collection.is_empty:
            return self.shaded_collection.n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        n_surfaces = self._illum_collection.n_surfaces \
            + self._shaded_collection.n_surfaces
        return n_surfaces

    @property
    def surface_indices(self):
        list_indices = []
        list_indices += self._illum_collection.surface_indices
        list_indices += self._shaded_collection.surface_indices
        return list_indices

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None):
        """Create a PVSegment from the coordinates of a single ``shapely``
        :py:class:`LineString`"""
        col = ShadeCollection.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector)
        # Realized that needed to instantiate other_col, otherwise could
        # end up with shared collection among different PV segments
        other_col = ShadeCollection(list_surfaces=[], shaded=not shaded)
        if shaded:
            return cls(illum_collection=other_col,
                       shaded_collection=col, index=index)
        else:
            return cls(illum_collection=col,
                       shaded_collection=other_col, index=index)

    @property
    def shaded_collection(self):
        return self._shaded_collection

    @shaded_collection.setter
    def shaded_collection(self, new_collection):
        assert new_collection.shaded, "surface should be shaded"
        self._shaded_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @shaded_collection.deleter
    def shaded_collection(self):
        self._shaded_collection = ShadeCollection(shaded=True)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def illum_collection(self):
        return self._illum_collection

    @illum_collection.setter
    def illum_collection(self, new_collection):
        assert not new_collection.shaded, "surface should not be shaded"
        self._illum_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @illum_collection.deleter
    def illum_collection(self):
        self._illum_collection = ShadeCollection(shaded=False)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def shaded_length(self):
        return self._shaded_collection.length

    @property
    def all_surfaces(self):
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self._illum_collection.list_surfaces
            self._all_surfaces += self._shaded_collection.list_surfaces
        return self._all_surfaces


class BaseSide(GeometryCollection):
    """A side represents a fixed collection of
    :py:class:`~pvfactors.pvsurfaces.PVSegment` objects that should
    all be collinear, with the same normal vector"""

    def __init__(self, list_segments=[]):
        """Create a side geometry."""
        check_collinear(list_segments)
        self.list_segments = tuple(list_segments)
        self._all_surfaces = None
        super(BaseSide, self).__init__(list_segments)

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, n_segments=1):
        if n_segments == 1:
            list_pvsegments = [PVSegment.from_linestring_coords(
                coords, shaded=shaded, normal_vector=normal_vector,
                index=index)]
        else:
            # Discretize coords and create segments accordingly
            linestring = LineString(coords)
            fractions = np.linspace(0., 1., num=n_segments + 1)
            list_points = [linestring.interpolate(fraction, normalized=True)
                           for fraction in fractions]
            list_pvsegments = []
            for idx in range(n_segments):
                new_coords = list_points[idx:idx + 2]
                pvsegment = PVSegment.from_linestring_coords(
                    new_coords, shaded=shaded, normal_vector=normal_vector,
                    index=index)
                list_pvsegments.append(pvsegment)
        return cls(list_segments=list_pvsegments)

    @property
    def n_vector(self):
        if len(self.list_segments):
            return self.list_segments[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def shaded_length(self):
        shaded_length = 0.
        for segment in self.list_segments:
            shaded_length += segment.shaded_length
        return shaded_length

    @property
    def n_surfaces(self):
        n_surfaces = 0
        for segment in self.list_segments:
            n_surfaces += segment.n_surfaces
        return n_surfaces

    @property
    def all_surfaces(self):
        if self._all_surfaces is None:
            self._all_surfaces = []
            for segment in self.list_segments:
                self._all_surfaces += segment.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        list_indices = []
        for seg in self.list_segments:
            list_indices += seg.surface_indices
        return list_indices

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Plot all PV segments in side object"""
        for segment in self.list_segments:
            segment.plot(ax, color_shaded=color_shaded,
                         color_illum=color_illum, with_index=with_index)

    def cast_shadow(self, linestring):
        """Cast designated linestring shadow on PV segments"""
        for segment in self.list_segments:
            segment.cast_shadow(linestring)

    def merge_shaded_areas(self):
        """Merge shaded areas of all pvsegments"""
        for seg in self.list_segments:
            seg._shaded_collection.merge_surfaces()

    def cut_at_point(self, point):
        """Cut side geometry at a given point, making sure that the point
        is contained by the side beforehand"""
        if contains(self, point):
            for segment in self.list_segments:
                # Nothing will happen to the segments that do not contain
                # the point
                segment.cut_at_point(point)


class BasePVArray(object):
    """Base class for PV arrays in pvfactors"""

    registry_cols = ['geom', 'line_type', 'pvrow_index', 'side',
                     'pvsegment_index', 'shaded', 'surface_index']

    def __init__(self, list_pvrows=[], ground=None, distance=None,
                 height=None):
        self.pvrows = list_pvrows
        self.ground = ground
        self.distance = distance
        self.height = height

        # Property related attributes: will not be built unless called
        self._all_surfaces = None
        self._surface_registry = None
        self._view_matrix = None
        self._obstr_matrix = None
        self._surfaces_indexed = False

    def plot(self, ax, with_index=False):
        """Plot PV array"""
        # Plot pv array structures
        self.ground.plot(ax, color_shaded=COLOR_DIC['ground_shaded'],
                         color_illum=COLOR_DIC['ground_illum'],
                         with_index=with_index)
        for pvrow in self.pvrows:
            pvrow.plot(ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                       color_illum=COLOR_DIC['pvrow_illum'],
                       with_index=with_index)

        # Plot formatting
        ax.axis('equal')
        if self.distance is not None:
            n_pvrows = len(self.pvrows)
            ax.set_xlim(- 0.5 * self.distance,
                        (n_pvrows - 0.5) * self.distance)
        if self.height is not None:
            ax.set_ylim(- self.height, 2 * self.height)
        ax.set_xlabel("x [m]", fontsize=PLOT_FONTSIZE)
        ax.set_ylabel("y [m]", fontsize=PLOT_FONTSIZE)

    @property
    def all_surfaces(self):
        if self._all_surfaces is None:
            list_surfaces = []
            list_surfaces += self.ground.all_surfaces
            for pvrow in self.pvrows:
                list_surfaces += pvrow.all_surfaces
            self._all_surfaces = list_surfaces
        return self._all_surfaces

    @property
    def n_surfaces(self):
        n_surfaces = 0
        n_surfaces += self.ground.n_surfaces
        for pvrow in self.pvrows:
            n_surfaces += pvrow.front.n_surfaces
            n_surfaces += pvrow.back.n_surfaces
        return n_surfaces

    @property
    def surface_registry(self):
        if self._surface_registry is None:
            self._surface_registry = self._build_surface_registry()
        return self._surface_registry

    @property
    def view_matrix(self):
        """How to build the view matrix will be specific to the pv array
        considered, so ``_build_view_matrix()`` needs to be implemented in
        the child class"""
        if self._view_matrix is None:
            self._view_matrix, self._obstr_matrix = self._build_view_matrix()
        return self._view_matrix

    @property
    def obstr_matrix(self):
        """Get obstruction matrix"""
        if self._obstr_matrix is None:
            self._view_matrix, self._obstr_matrix = self._build_view_matrix()
        return self._obstr_matrix

    @property
    def surface_indices(self):
        list_indices = []
        list_indices += self.ground.surface_indices
        for pvrow in self.pvrows:
            list_indices += pvrow.surface_indices
        return list_indices

    def index_all_surfaces(self):
        """Add unique indices to all surfaces"""
        for idx, surface in enumerate(self.all_surfaces):
            surface.index = idx
        self._surfaces_indexed = True

    def _build_view_matrix(self):
        raise NotImplementedError

    def _build_surface_registry(self):
        dict_registry = {k: [] for k in self.registry_cols}
        # Fill up registry with ground surfaces
        surf_line_type = 'ground'
        pvrow_index = np.nan
        side = np.nan
        for idx_seg, gnd_seg in enumerate(self.ground.list_segments):
            pvsegment_index = idx_seg
            # Illuminated collection
            shaded = False
            for surf in gnd_seg.illum_collection.list_surfaces:
                dict_registry['line_type'].append(surf_line_type)
                dict_registry['pvsegment_index'].append(pvsegment_index)
                dict_registry['pvrow_index'].append(pvrow_index)
                dict_registry['side'].append(side)
                dict_registry['shaded'].append(shaded)
                dict_registry['geom'].append(surf)
                dict_registry['surface_index'].append(surf.index)
            shaded = True
            for surf in gnd_seg.shaded_collection.list_surfaces:
                dict_registry['line_type'].append(surf_line_type)
                dict_registry['pvsegment_index'].append(pvsegment_index)
                dict_registry['pvrow_index'].append(pvrow_index)
                dict_registry['side'].append(side)
                dict_registry['shaded'].append(shaded)
                dict_registry['geom'].append(surf)
                dict_registry['surface_index'].append(surf.index)
        # Fill up registry with pvrow surfaces
        surf_line_type = 'pvrow'
        for idx_pvrow, pvrow in enumerate(self.pvrows):
            pvrow_index = idx_pvrow
            # Front side
            side = 'front'
            for idx_seg, seg in enumerate(pvrow.front.list_segments):
                pvsegment_index = idx_seg
                # Illuminated
                shaded = False
                for surf in seg.illum_collection.list_surfaces:
                    dict_registry['line_type'].append(surf_line_type)
                    dict_registry['pvsegment_index'].append(pvsegment_index)
                    dict_registry['pvrow_index'].append(pvrow_index)
                    dict_registry['side'].append(side)
                    dict_registry['shaded'].append(shaded)
                    dict_registry['geom'].append(surf)
                    dict_registry['surface_index'].append(surf.index)
                # Shaded
                shaded = True
                for surf in seg.shaded_collection.list_surfaces:
                    dict_registry['line_type'].append(surf_line_type)
                    dict_registry['pvsegment_index'].append(pvsegment_index)
                    dict_registry['pvrow_index'].append(pvrow_index)
                    dict_registry['side'].append(side)
                    dict_registry['shaded'].append(shaded)
                    dict_registry['geom'].append(surf)
                    dict_registry['surface_index'].append(surf.index)
            # Back side
            side = 'back'
            for idx_seg, seg in enumerate(pvrow.back.list_segments):
                pvsegment_index = idx_seg
                # Illuminated
                shaded = False
                for surf in seg.illum_collection.list_surfaces:
                    dict_registry['line_type'].append(surf_line_type)
                    dict_registry['pvsegment_index'].append(pvsegment_index)
                    dict_registry['pvrow_index'].append(pvrow_index)
                    dict_registry['side'].append(side)
                    dict_registry['shaded'].append(shaded)
                    dict_registry['geom'].append(surf)
                    dict_registry['surface_index'].append(surf.index)
                # Shaded
                shaded = True
                for surf in seg.shaded_collection.list_surfaces:
                    dict_registry['line_type'].append(surf_line_type)
                    dict_registry['pvsegment_index'].append(pvsegment_index)
                    dict_registry['pvrow_index'].append(pvrow_index)
                    dict_registry['side'].append(side)
                    dict_registry['shaded'].append(shaded)
                    dict_registry['geom'].append(surf)
                    dict_registry['surface_index'].append(surf.index)

        # Make dataframe
        surface_registry = pd.DataFrame.from_dict(dict_registry)
        return surface_registry
