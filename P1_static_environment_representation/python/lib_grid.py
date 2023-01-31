# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library for Grid specific classes and functions
# =====================================================================================================================

import numpy as np
import lib_const

class grid_properties:
    """ Create a grid and the relevant helper functions """
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

        self.num_cells_x = int( np.ceil( ( self.max_x + lib_const.eps - self.min_x ) / self.dx ) )
        self.num_cells_y = int( np.ceil( ( self.max_y + lib_const.eps - self.min_y ) / self.dy ) )
        self.num_cells = self.num_cells_x * self.num_cells_y

        self.x_ids = np.arange(0, self.num_cells_x, dtype=np.uint32)
        self.y_ids = np.arange(0, self.num_cells_y, dtype=np.uint32)
        self.cell_ids = np.arange(0, self.num_cells, dtype=np.uint32)

        x_coord, y_coord \
            = self.compute_xy_coordinates_from_xy_ids(self.x_ids, self.y_ids)
        self.x_coord = x_coord
        self.y_coord = y_coord

    def compute_xy_coordinates_from_xy_ids(self, x_ids, y_ids):
        x_coord = ( x_ids.astype(np.float32) + 0.5 ) * self.dx + self.min_x
        y_coord = ( y_ids.astype(np.float32) + 0.5 ) * self.dy + self.min_y
        return x_coord, y_coord

    def compute_xy_ids_from_xy_coordinates(self, meas_x, meas_y):
        cell_x_idx = np.floor( ( meas_x - self.min_x ) / self.dx )
        cell_y_idx = np.floor( ( meas_y - self.min_y ) / self.dy )
        return cell_x_idx.astype(np.uint32), cell_y_idx.astype(np.uint32)

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

def roundoff_azimuth_degree_within_correct_range(theta):
    """ theta is coorcted such that the value always lies between [-180, 180] """
    abs_theta =  np.abs(theta)
    val_theta = -np.sign(theta) * (360 - abs_theta)
    theta = np.where(abs_theta > 180, val_theta, theta)
    return theta


def compute_azimuth_res(
    theta,
    theta_res_min, 
    theta_res_max, 
    theta_min, 
    theta_max):
    """ compute angular cell resolutions for variable sized polar grid ( linear variation )
    Input: theta - angle in degree
         : theta_res_min, theta_res_max - min and max angular resolution that the grid can have in degree
         : theta_min, theta_max - min and max angle coverage that the grid has
    Output: theta_res - calculated angular resolution in degree
    """
    theta = np.abs(roundoff_azimuth_degree_within_correct_range(theta))
    if theta >= 0 and theta <= 90: quadrant = 14
    elif theta > 90 and theta <= 180: quadrant = 23
    if quadrant == 23: theta = 180 - theta
    theta_res = ( ( theta - theta_min ) / ( theta_max - theta_min ) ) * ( theta_res_max - theta_res_min ) + theta_res_min
    return theta_res


def compute_azimuth_grid_cell_boundary(
    theta, 
    theta_res):
    """ compute the boundary value of a cell given the azimuth angle in degree
    The value have been adjusted such that angles are always between [-180, 180]
    Input: theta - angle in degree
         : theta_res - the corrosponding angle resolution in degree
    Output: theta_lower, theta_upper - the grid cell bounds in degree
    """
    theta_lower = roundoff_azimuth_degree_within_correct_range(theta - 0.5 * theta_res)
    theta_upper = roundoff_azimuth_degree_within_correct_range(theta + 0.5 * theta_res)
    return theta_lower, theta_upper


class polar_grid_properties:
    """ Create a polar grid with variable sized cell resolutions and the relevant helper functions """
    def __init__(
        self, 
        min_range, 
        max_range, 
        min_azimuth, 
        max_azimuth, 
        range_res, 
        azimuth_res,
        theta_res_min, 
        theta_res_max):

        self.min_range = min_range
        self.max_range = max_range

        a = roundoff_azimuth_degree_within_correct_range(min_azimuth)
        b = roundoff_azimuth_degree_within_correct_range(max_azimuth)
        if a >= b :
            self.min_azimuth = b
            self.max_azimuth = a
        else:
            self.min_azimuth = a
            self.max_azimuth = b
        
        self.range_res = range_res
        self.azimuth_res = azimuth_res

        self.num_cells_range = int( np.ceil( ( self.max_range + lib_const.eps - self.min_range ) / self.range_res ) )
        self.num_cells_azimuth = int( np.ceil( ( self.max_azimuth + lib_const.eps - self.min_azimuth ) / self.azimuth_res ) )
        self.num_cells = self.num_cells_range * self.num_cells_azimuth

        self.range_ids = np.arange(0, self.num_cells_range, dtype=np.uint32)
        self.azimuth_ids = np.arange(0, self.num_cells_azimuth, dtype=np.uint32)
        self.cell_ids = np.arange(0, self.num_cells, dtype=np.uint32)

        range_coord, azimuth_coord \
            = self.compute_ra_coordinates_from_ra_ids(self.range_ids, self.azimuth_ids)
        self.range_coord = range_coord
        self.azimuth_coord = azimuth_coord
        
        self.theta_res_min = theta_res_min
        self.theta_res_max = theta_res_max
        compute_azimuth_res_vfunc = np.vectorize(compute_azimuth_res)
        self.theta_res = compute_azimuth_res_vfunc(self.azimuth_coord, self.theta_res_min,self.theta_res_max, 0, 90)
        self.theta_lower, self.theta_upper = compute_azimuth_grid_cell_boundary(self.azimuth_coord, self.theta_res)
        self.azimuth_coord = roundoff_azimuth_degree_within_correct_range(self.azimuth_coord)

    def compute_ra_coordinates_from_ra_ids(self, range_ids, azimuth_ids):
        range_coord = ( range_ids.astype(np.float32) + 0.5 ) * self.range_res + self.min_range
        azimuth_coord = ( azimuth_ids.astype(np.float32) + 0.5 ) * self.azimuth_res + self.min_azimuth
        azimuth_coord = roundoff_azimuth_degree_within_correct_range(azimuth_coord)
        return range_coord, azimuth_coord

    def compute_ra_ids_from_ra_coordinates(self, ranges, azimuths):
        cell_range_idx = np.floor( ( ranges - self.min_range ) / self.range_res )
        cell_azimuith_idx = np.floor( ( azimuths - self.min_azimuth ) / self.azimuth_res )
        return cell_range_idx.astype(np.uint32), cell_azimuith_idx.astype(np.uint32)

# ====================================================================================================================

class FOV_grid_coverage_sensor_frame:
    """ Identifies different sensor fov zones that are getting intersected within the grid """
    def __init__(self, cart_grid_properties):
        self.compute_fov_coverage(cart_grid_properties)

    def compute_fov_coverage(self, cart_grid_properties):
        self.x_coord, self.y_coord \
            = cart_grid_properties.compute_xy_coordinates_from_scalar_ids(cart_grid_properties.cell_ids)
        self.fov_coverage_flag \
            = identify_coordinates_that_are_within_sensor_fov(
                self.x_coord, self.y_coord, lib_const.radar_fov_ranges, lib_const.radar_fov_azimuths)


class FOV_grid_coverage:
    """ Identifies different sensor fov zones that are getting intersected within the grid """
    def __init__(self, cart_grid_properties):
        self.fov_coverage_flag = np.zeros((lib_const.num_radars, cart_grid_properties.num_cells), dtype=np.bool8)
        self.compute_fov_coverage(cart_grid_properties)

    def compute_fov_coverage(self, cart_grid_properties):
        self.x_coord, self.y_coord \
            = cart_grid_properties.compute_xy_coordinates_from_scalar_ids(cart_grid_properties.cell_ids)
        for sensor in range(lib_const.num_radars):
            mount = lib_const.radars_mount[sensor]
            x, y = \
                coordinate_transform_px_py_vf_to_sf(
                    self.x_coord, self.y_coord, 
                    mount[0], mount[1], mount[2])
            self.fov_coverage_flag[sensor, :] \
                = identify_coordinates_that_are_within_sensor_fov(
                    x, y, lib_const.radar_fov_ranges, lib_const.radar_fov_azimuths)

# ====================================================================================================================

def convert_grid_cart_to_polar(xcoord, ycoord):
    range_coord = np.sqrt( xcoord**2 + ycoord**2 )
    azi_coord = np.arctan2(ycoord, xcoord)
    azi_coord = lib_const.rad2deg * azi_coord
    return range_coord, azi_coord


def convert_grid_polar_to_cartesian(range, azimuth):
    """ convert meas polar to cartesian """
    azimuth = lib_const.deg2rad * azimuth
    x = range * np.cos(azimuth)
    y = range * np.sin(azimuth)
    return x, y


def convert_meas_polar_to_cartesian(range, azimuth):
    """ convert meas polar to cartesian """
    x = range * np.cos(azimuth)
    y = range * np.sin(azimuth)
    return x, y


def convert_meas_cartesian_to_polar(xcoord, ycoord):
    """ convert meas cartesian to polar """
    range = np.sqrt( xcoord ** 2 + ycoord ** 2 )
    azimuth = np.arctan2(ycoord, xcoord)
    return range, azimuth


def coordinate_transform_px_py_vf_to_sf(px, py, tx, ty, theta):
    """ coordinate transform measurements from vehicle frame to sensor frame """
    px = px - tx
    py = py - ty
    px_cts = px * np.cos(-theta) - py * np.sin(-theta)
    py_cts = px * np.sin(-theta) + py * np.cos(-theta)
    return px_cts, py_cts 

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

# =====================================================================================================================

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


def select_cells_within_the_grid_boundary(
    cellidx_x,
    cellidx_y,
    grid_num_rows,
    grid_num_cols):
    """ given grid indexes identify which grid cells are within the defined grid """
    condition = np.logical_and( 
        np.logical_and( cellidx_x < grid_num_rows, cellidx_x >= 0 ), 
        np.logical_and( cellidx_y < grid_num_cols, cellidx_y >= 0 ) )
    return condition


def select_points_within_polar_roi(
    log_odds_grid_prop, 
    polar_grid_prop, 
    cellids):
    """ select points from log-odds grid such that they lie within the defined polar grid
    Input: log_odds_grid_prop - log-odds grid object which contains the properties of the log-odds measurement grid
         : polar_grid_prop - polar grid object which contains the properties of the polar grid
         : cellids - valid / tracked log odds cell ids
    Output: rangecoord, azimuthcoord - selected range and azimuth measurement/object coordinates in the defined polar grid
    """
    xcoord, ycoord = log_odds_grid_prop.compute_xy_coordinates_from_scalar_ids(cellids)
    rangecoord, azimuthcoord = convert_grid_cart_to_polar(xcoord, ycoord)
    condition = select_points_within_grid_boundary(
        rangecoord, azimuthcoord, 
        polar_grid_prop.min_range, polar_grid_prop.max_range,
        polar_grid_prop.min_azimuth, polar_grid_prop.max_azimuth)
    rangecoord = rangecoord[condition]
    azimuthcoord = azimuthcoord[condition]
    return rangecoord, azimuthcoord

# =====================================================================================================================

def map_cartesian_grid_ids_to_polar_grid(
    cart_grid_prop, 
    polar_grid_prop):
    """ compute a list of cartesian grid cell ids which can be mapped to the polar grid cell ids
    Input: cart_grid_obj - cartesian grid object which contains the properties of the cartesian grid
         : polar_grid_obj - polar grid object which contains the properties of the polar grid
    Output: cart_cell_ids - mapped cartesian cell ids
          : rids, aids - corrosponding ids to polar grid
    """
    xcoord, ycoord = cart_grid_prop.compute_xy_coordinates_from_scalar_ids(cart_grid_prop.cell_ids)
    rcoord, acoord = convert_grid_cart_to_polar(xcoord, ycoord)
    condition = select_points_within_grid_boundary(
        rcoord, acoord, 
        polar_grid_prop.min_range, polar_grid_prop.max_range,
        polar_grid_prop.min_azimuth, polar_grid_prop.max_azimuth)
    rids, aids = polar_grid_prop.compute_ra_ids_from_ra_coordinates(rcoord[condition], acoord[condition])
    cart_cell_ids = cart_grid_prop.cell_ids[condition]
    return cart_cell_ids, rids, aids


    