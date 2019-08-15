"""Timeseries geometry classes"""

import numpy as np
from pvlib.tools import cosd, sind
from pvfactors.config import DISTANCE_TOLERANCE, COLOR_DIC
from pvfactors.geometry.base import PVSurface
from shapely.geometry import GeometryCollection


class TsPVRow(object):

    def __init__(self, ts_front_side, ts_back_side, xy_center):
        self.front = ts_front_side
        self.back = ts_back_side
        self.xy_center = xy_center

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec,
                        cut, shaded_length_front, shaded_length_back,
                        is_left_edge, is_right_edge):
        # Calculate front side coords
        ts_front = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('front', 1),
            shaded_length_front, is_left_edge,
            is_right_edge)
        # Calculate back side coords
        ts_back = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('back', 1),
            shaded_length_front, is_left_edge,
            is_right_edge)

        return cls(ts_front, ts_back, xy_center)

    def surfaces_at_idx(self, idx):
        list_surfaces = []
        list_surfaces.append(self.front.surfaces_at_idx(idx))
        list_surfaces.append(self.back.surfaces_at_idx(idx))
        return list_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        self.front.plot_at_idx(idx, ax, color_shaded=color_shaded,
                               color_illum=color_illum)
        self.back.plot_at_idx(idx, ax, color_shaded=color_shaded,
                              color_illum=color_illum)


class TsSide(object):

    def __init__(self, segments):
        self.list_segments = segments

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec, cut,
                        shaded_length, is_left_edge, is_right_edge):
        # Create surface coords
        # Put surfaces in shaded collections
        # Put collections into segment coords
        # Put shaded collection in side coords

        # Create Ts segments
        x_center, y_center = xy_center
        radius = width / 2.
        segment_length = width / cut
        list_segments = []
        for i in range(cut):
            r1 = radius - i * segment_length
            r2 = radius - (i + 1) * segment_length
            x1 = r1 * cosd(rotation_vec + 180.) + x_center
            y1 = r1 * sind(rotation_vec + 180.) + y_center
            x2 = r2 * cosd(rotation_vec + 180) + x_center
            y2 = r2 * sind(rotation_vec + 180) + y_center
            segment_coords = TsLineCoords.from_array(
                np.array([[x1, y1], [x2, y2]]))
            illum = TsSurface(segment_coords)
            shaded = TsSurface(TsLineCoords.from_array(
                np.array([[x1, y1], [x1, y1]])))
            segment = TsDualSegment(segment_coords, illum, shaded)
            list_segments.append(segment)

        return cls(list_segments)

    def surfaces_at_idx(self, idx):
        list_surfaces = []
        for segment in self.list_segments:
            list_surfaces.append(segment.surfaces_at_idx(idx))
        return list_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        for segment in self.list_segments:
            segment.plot_at_idx(idx, ax, color_shaded=color_shaded,
                                color_illum=color_illum)

    @property
    def shaded_length(self):
        length = 0.
        for seg in self.list_segments:
            length += seg.shaded.length
        return length


class TsDualSegment(object):
    """A TsDualSegment is a timeseries segment that can only have
    1 shaded surface and 1 illuminated surface. This allows consistent
    indexing of the object."""

    def __init__(self, coords, illum_ts_surface, shaded_ts_surface):
        self.coords = coords
        self.illum = illum_ts_surface
        self.shaded = shaded_ts_surface

    def surfaces_at_idx(self, idx):
        list_surfaces = []
        illum_surface = self.illum.at(idx)
        shaded_surface = self.shaded.at(idx)
        if not illum_surface.is_empty:
            list_surfaces.append(illum_surface)
        if not shaded_surface.is_empty:
            list_surfaces.append(shaded_surface)
        return list_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        self.illum.plot_at_idx(idx, ax, color_illum)
        self.shaded.plot_at_idx(idx, ax, color_shaded)

    @property
    def length(self):
        return self.illum.length + self.shaded.length


class TsSurface(object):

    def __init__(self, coords, shaded=None):
        self.coords = coords
        self.length = np.sqrt((coords.b2.y - coords.b1.y)**2
                              + (coords.b2.x - coords.b1.x)**2)

    def at(self, idx):
        if self.length[idx] < DISTANCE_TOLERANCE:
            return GeometryCollection()
        else:
            return PVSurface(self.coords.at(idx))

    def plot_at_idx(self, idx, ax, color):
        if self.length[idx] > DISTANCE_TOLERANCE:
            self.at(idx).plot(ax, color=color)


class TsLineCoords(object):

    def __init__(self, b1_ts_coords, b2_ts_coords, coords=None):
        self.b1 = b1_ts_coords
        self.b2 = b2_ts_coords
        if coords is None:
            self.coords = np.array([b1_ts_coords, b2_ts_coords])
        else:
            self.coords = coords

    def at(self, idx):
        return self.coords[:, :, idx]

    @classmethod
    def from_array(cls, coords_array):
        b1 = TsPointCoords(coords_array[0, :, :])
        b2 = TsPointCoords(coords_array[1, :, :])
        return cls(b1, b2, coords=coords_array)


class TsPointCoords(object):

    def __init__(self, coords):
        self.x = coords[0, :]
        self.y = coords[1, :]
        self.coords = coords

    def at(self, idx):
        return self.coords[:, idx]
