# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library for Grid specific classes and methods
# =====================================================================================================================
import numpy as np
import const
from lib_read_data import extract_radar_mount_info


class grid_properties:
    """ Create a grid and the relevant helper methods """
    def __init__(
        self, 
        min_x, 
        max_x, 
        min_y, 
        max_y, 
        dx, 
        dy):

        self.min_x = min_x
        self.max_x = max_x
        
        self.min_y = min_y
        self.max_y = max_y
        
        self.dx = dx
        self.dy = dy

        self.num_cells_x = int( np.ceil( ( self.max_x + const.eps - self.min_x ) / self.dx ) )
        self.num_cells_y = int( np.ceil( ( self.max_y + const.eps - self.min_y ) / self.dy ) )
        self.num_cells = self.num_cells_x * self.num_cells_y

    def compute_xy_coordinates_from_xy_ids(self, x_ids, y_ids):
        x_coord = ( x_ids.astype(np.float32) + 0.5 ) * self.dx + self.min_x
        y_coord = ( y_ids.astype(np.float32) + 0.5 ) * self.dy + self.min_y
        return x_coord, y_coord

    def compute_xy_ids_from_xy_coordinates(self, meas_x, meas_y):
        cell_x_idx = np.floor( ( meas_x - self.min_x ) / self.dx )
        cell_y_idx = np.floor( ( meas_y - self.min_y ) / self.dy )
        return cell_x_idx.astype(np.int32), cell_y_idx.astype(np.int32)

    def compute_scalar_ids_from_xy_ids(self, x_ids, y_ids):
        scalar_ids = x_ids * self.num_cells_y + y_ids
        return scalar_ids

    def compute_xy_ids_from_scalar_ids(self, scalar_ids):
        x_ids = scalar_ids // self.num_cells_y
        y_ids = scalar_ids - x_ids * self.num_cells_y
        return x_ids, y_ids

    def compute_scalar_ids_from_xy_coordinates(self, meas_x, meas_y):
        x_ids, y_ids = self.compute_xy_ids_from_xy_coordinates(meas_x, meas_y)
        scalar_ids = self.compute_scalar_ids_from_xy_ids(x_ids, y_ids)
        return scalar_ids

    def compute_xy_coordinates_from_scalar_ids(self, scalar_ids):
        x_ids, y_ids = self.compute_xy_ids_from_scalar_ids(scalar_ids)
        x_coord, y_coord = self.compute_xy_coordinates_from_xy_ids(x_ids, y_ids)
        return x_coord, y_coord

# ====================================================================================================================

class FOV_grid_coverage_sensor_frame:
    """ Identifies different sensor fov zones that are getting intersected within the grid """
    def __init__(self, cart_grid_properties):
        self.fov_coverage_flag = self.compute_fov_coverage(cart_grid_properties)

    def compute_fov_coverage(self, cart_grid_properties):
        cell_ids = np.arange(0, cart_grid_properties.num_cells, dtype=np.uint32)
        x_coord, y_coord = cart_grid_properties.compute_xy_coordinates_from_scalar_ids(cell_ids)
        return identify_coordinates_that_are_within_sensor_fov(x_coord, y_coord, const.radar_fov_ranges, const.radar_fov_azimuths)


class FOV_grid_coverage:
    """ Identifies different sensor fov zones that are getting intersected within the grid """
    def __init__(self, cart_grid_properties, scene_name):
        self.mount_parameters = extract_radar_mount_info(scene_name)
        self.fov_coverage_flag = np.zeros((const.num_radars, cart_grid_properties.num_cells), dtype=np.bool8)
        self.compute_fov_coverage(cart_grid_properties)

    def compute_fov_coverage(self, cart_grid_properties):
        cell_ids = np.arange(0, cart_grid_properties.num_cells, dtype=np.uint32)
        x_coord, y_coord = cart_grid_properties.compute_xy_coordinates_from_scalar_ids(cell_ids)
        for sensor in range(const.num_radars):
            mount = self.mount_parameters[sensor]
            x, y = coordinate_transform_px_py_vf_to_sf(x_coord, y_coord, mount[0], mount[1], mount[2])
            self.fov_coverage_flag[sensor, :] = identify_coordinates_that_are_within_sensor_fov(x, y, const.radar_fov_ranges, const.radar_fov_azimuths)

# ====================================================================================================================

def identify_coordinates_that_are_within_sensor_fov(
    x_coord, y_coord, 
    sensor_fov_ranges, 
    sensor_fov_azimuths):

    """ This funcion returns a boolean flag indicating which coordinates are within the sensor fov """
    sorted_idx = np.argsort(sensor_fov_azimuths)
    sensor_fov_azimuths = sensor_fov_azimuths[sorted_idx]
    sensor_fov_ranges = sensor_fov_ranges[sorted_idx]

    #cartesian to polar conversion
    r, th = convert_grid_cart_to_polar(x_coord, y_coord)

    condition1 = np.logical_and( 
        th >= sensor_fov_azimuths[0], 
        th <= sensor_fov_azimuths[-1] )
    condition2 = ( r <= np.max(sensor_fov_ranges) )
    condition1 = np.logical_and( condition1, condition2 )

    selected_azimuth = th[condition1]
    selected_range = r[condition1]

    # compute boundary idx
    fov_azimuths = np.expand_dims(sensor_fov_azimuths, axis=-1)
    fov_azimuths = np.repeat(fov_azimuths, selected_azimuth.shape[0], axis=-1)

    selected_azimuth_expd = np.expand_dims(selected_azimuth, axis=0)
    flag = selected_azimuth_expd >= fov_azimuths

    # extract boundary idx
    bounday_idx_upper = np.sum(flag, axis=0)
    bounday_idx_lower = bounday_idx_upper - 1

    # extract azimuth
    azimuth_upper = sensor_fov_azimuths[bounday_idx_upper]
    azimuth_lower = sensor_fov_azimuths[bounday_idx_lower]

    # extract ranges
    ranges_upper = sensor_fov_ranges[bounday_idx_upper]
    ranges_lower = sensor_fov_ranges[bounday_idx_lower]

    # interpolate and identify range within boundary
    r_interp =  ( selected_azimuth - azimuth_lower ) / ( azimuth_upper - azimuth_lower ) * ( ranges_upper - ranges_lower ) + ranges_lower
    condition2 = selected_range <= r_interp

    meas_idx = np.arange(x_coord.shape[0])[condition1]
    condition1[meas_idx] = condition2
    return condition1

# ====================================================================================================================

def convert_grid_cart_to_polar(xcoord, ycoord):
    range_coord = np.sqrt( xcoord**2 + ycoord**2 )
    azi_coord = np.arctan2(ycoord, xcoord)
    azi_coord = const.rad2deg * azi_coord
    return range_coord, azi_coord


def coordinate_transform_px_py_vf_to_sf(px, py, tx, ty, theta):
    """ coordinate transform measurements from vehicle frame to sensor frame """
    px = px - tx
    py = py - ty
    px_cts = px * np.cos(-theta) - py * np.sin(-theta)
    py_cts = px * np.sin(-theta) + py * np.cos(-theta)
    return px_cts, py_cts

# ====================================================================================================================

def select_points_within_grid_boundary(
    query_x, 
    query_y, 
    grid_min_x, 
    grid_max_x,
    grid_min_y, 
    grid_max_y):
    """ given points (x,y) in meters indentify which points are within the defined grid """
    condition = np.logical_and(
        np.logical_and(
            query_x <= grid_max_x, 
            query_x >= grid_min_x), 
        np.logical_and(
            query_y <= grid_max_y, 
            query_y >= grid_min_y))
    return condition

# ====================================================================================================================