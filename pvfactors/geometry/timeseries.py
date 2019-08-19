"""Timeseries geometry classes"""

import numpy as np
from pvlib.tools import cosd, sind
from pvfactors.config import DISTANCE_TOLERANCE, COLOR_DIC
from pvfactors.geometry.base import (
    PVSurface, ShadeCollection, PVSegment, BaseSide)
from pvfactors.geometry.pvrow import PVRow
from shapely.geometry import GeometryCollection


class TsPVRow(object):

    def __init__(self, ts_front_side, ts_back_side, xy_center, index=None,
                 full_pvrow_coords=None):
        self.front = ts_front_side
        self.back = ts_back_side
        self.xy_center = xy_center
        self.index = index
        self.full_pvrow_coords = full_pvrow_coords

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec,
                        cut, shaded_length_front, shaded_length_back,
                        index=None):
        """Shading will always be zero when pv rows are flat"""
        # Calculate full pvrow coords
        pvrow_coords = TsPVRow._calculate_full_coords(
            xy_center, width, rotation_vec)
        # Calculate normal vectors
        dx = pvrow_coords.b2.x - pvrow_coords.b1.x
        dy = pvrow_coords.b2.y - pvrow_coords.b1.y
        normal_vec_front = np.array([-dy, dx])
        # Calculate front side coords
        ts_front = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('front', 1),
            shaded_length_front, n_vector=normal_vec_front)
        # Calculate back side coords
        ts_back = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('back', 1),
            shaded_length_back, n_vector=-normal_vec_front)

        return cls(ts_front, ts_back, xy_center, index=index,
                   full_pvrow_coords=pvrow_coords)

    @staticmethod
    def _calculate_full_coords(xy_center, width, rotation):
        x_center, y_center = xy_center
        radius = width / 2.
        # Calculate coords
        x1 = radius * cosd(rotation + 180.) + x_center
        y1 = radius * sind(rotation + 180.) + y_center
        x2 = radius * cosd(rotation) + x_center
        y2 = radius * sind(rotation) + y_center
        coords = TsLineCoords.from_array(np.array([[x1, y1], [x2, y2]]))
        return coords

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

    def at(self, idx):
        front_geom = self.front.at(idx)
        back_geom = self.back.at(idx)
        pvrow = PVRow(front_side=front_geom, back_side=back_geom,
                      index=self.index, original_linestring=None)
        return pvrow


class TsSide(object):

    def __init__(self, segments, n_vector=None):
        self.list_segments = segments
        self.n_vector = n_vector

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec, cut,
                        shaded_length, n_vector=None):
        """
        Shading will always be zero when PV rows are flat
        """

        mask_tilted_to_left = rotation_vec >= 0

        # Create Ts segments
        x_center, y_center = xy_center
        radius = width / 2.
        segment_length = width / cut
        is_not_flat = rotation_vec != 0.

        # Calculate coords of shading point
        r_shade = radius - shaded_length
        x_sh = np.where(
            mask_tilted_to_left,
            r_shade * cosd(rotation_vec + 180.) + x_center,
            r_shade * cosd(rotation_vec) + x_center)
        y_sh = np.where(
            mask_tilted_to_left,
            r_shade * sind(rotation_vec + 180.) + y_center,
            r_shade * sind(rotation_vec) + y_center)

        # Calculate coords
        list_segments = []
        for i in range(cut):
            # Calculate segment coords
            r1 = radius - i * segment_length
            r2 = radius - (i + 1) * segment_length
            x1 = r1 * cosd(rotation_vec + 180.) + x_center
            y1 = r1 * sind(rotation_vec + 180.) + y_center
            x2 = r2 * cosd(rotation_vec + 180) + x_center
            y2 = r2 * sind(rotation_vec + 180) + y_center
            segment_coords = TsLineCoords.from_array(
                np.array([[x1, y1], [x2, y2]]))
            # Determine lowest and highest points of segment
            x_highest = np.where(mask_tilted_to_left, x2, x1)
            y_highest = np.where(mask_tilted_to_left, y2, y1)
            x_lowest = np.where(mask_tilted_to_left, x1, x2)
            y_lowest = np.where(mask_tilted_to_left, y1, y2)
            # Calculate illum and shaded coords
            x2_illum, y2_illum = x_highest, y_highest
            x1_shaded, y1_shaded, x2_shaded, y2_shaded = \
                x_lowest, y_lowest, x_lowest, y_lowest
            mask_all_shaded = (y_sh > y_highest) & (is_not_flat)
            mask_partial_shaded = (y_sh > y_lowest) & (~ mask_all_shaded) \
                & (is_not_flat)
            # Calculate second boundary point of shade
            x2_shaded = np.where(mask_all_shaded, x_highest, x2_shaded)
            x2_shaded = np.where(mask_partial_shaded, x_sh, x2_shaded)
            y2_shaded = np.where(mask_all_shaded, y_highest, y2_shaded)
            y2_shaded = np.where(mask_partial_shaded, y_sh, y2_shaded)
            x1_illum = x2_shaded
            y1_illum = y2_shaded
            illum_coords = TsLineCoords.from_array(
                np.array([[x1_illum, y1_illum], [x2_illum, y2_illum]]))
            shaded_coords = TsLineCoords.from_array(
                np.array([[x1_shaded, y1_shaded], [x2_shaded, y2_shaded]]))
            # Create illuminated and shaded surfaces
            illum = TsSurface(illum_coords, n_vector=n_vector)
            shaded = TsSurface(shaded_coords, n_vector=n_vector)
            # Create dual segment
            segment = TsDualSegment(segment_coords, illum, shaded,
                                    n_vector=n_vector)
            list_segments.append(segment)

        return cls(list_segments, n_vector=n_vector)

    def surfaces_at_idx(self, idx):
        list_surfaces = []
        for segment in self.list_segments:
            list_surfaces.append(segment.surfaces_at_idx(idx))
        return list_surfaces

    def at(self, idx):
        list_geom_segments = []
        for ts_seg in self.list_segments:
            list_geom_segments.append(ts_seg.at(idx))
        side = BaseSide(list_geom_segments)
        return side

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

    def __init__(self, coords, illum_ts_surface, shaded_ts_surface,
                 index=None, n_vector=None):
        self.coords = coords
        self.illum = illum_ts_surface
        self.shaded = shaded_ts_surface
        self.index = index
        self.n_vector = n_vector

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

    def at(self, idx):
        # Create illum collection
        illum_surface = self.illum.at(idx, shaded=False)
        list_illum_surfaces = [] if illum_surface.is_empty \
            else [illum_surface]
        illum_collection = ShadeCollection(
            list_surfaces=list_illum_surfaces, shaded=False,
            surface_params=None)
        # Create shaded collection
        shaded_surface = self.shaded.at(idx, shaded=True)
        list_shaded_surfaces = [] if shaded_surface.is_empty \
            else [shaded_surface]
        shaded_collection = ShadeCollection(
            list_surfaces=list_shaded_surfaces, shaded=True,
            surface_params=None)
        # Create PV segment
        segment = PVSegment(illum_collection=illum_collection,
                            shaded_collection=shaded_collection,
                            index=self.index)
        return segment

    @property
    def length(self):
        return self.illum.length + self.shaded.length

    @property
    def shaded_length(self):
        return self.shaded.length


class TsSurface(object):

    def __init__(self, coords, n_vector=None):
        self.coords = coords
        self.length = np.sqrt((coords.b2.y - coords.b1.y)**2
                              + (coords.b2.x - coords.b1.x)**2)
        self.n_vector = n_vector

    def at(self, idx, shaded=None):
        if self.length[idx] < DISTANCE_TOLERANCE:
            # return an empty geometry
            return GeometryCollection()
        else:
            # Get normal vector at that time
            n_vector = (self.n_vector[:, idx] if self.n_vector is not None
                        else None)
            # Return a pv surface geometry
            return PVSurface(self.coords.at(idx), shaded=shaded,
                             normal_vector=n_vector)

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
