# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library to run some use cases : road boundary and free space
# =====================================================================================================================

import numpy as np
from lib_log_odds_filter import filter_log_odds_by_thresholding
from lib_log_odds import compute_prob_from_log_odds
from lib_grid import (
    convert_grid_cart_to_polar, 
    convert_grid_polar_to_cartesian,
    select_points_within_polar_roi,
    map_cartesian_grid_ids_to_polar_grid,
    select_points_within_grid_boundary
)   


def popoulate_free_space_polar_grid_v1(
    polar_grid_prop, 
    occu_map_polar, 
    rangecoord, 
    azimuthcoord):
    """ polulate the free space probabilities in polar grid given the measurements in polar coordinate frame
    Input: polar_grid_prop - polar grid object which contains the properties of the polar grid
         : occu_map_polar - polar occupancy grid
         : rangecoord - measurement range 
         : azimuthcoord - measurement azimuth angle
    Output : occu_map_polar - polar grid containing the free space probabilities
    """
    rangeids, azimuthids = polar_grid_prop.compute_ra_ids_from_ra_coordinates(rangecoord, azimuthcoord)
    for i in range(polar_grid_prop.num_cells_azimuth):
        flag = ( azimuthids == i )
        cond = np.sum(flag.astype(np.int32))
        if cond == 0: occu_map_polar[:, i] = 0.02
        elif cond > 0:
            sel_ranges = rangecoord[flag]
            sel_range_idx = rangeids[flag]
            closest_range_idx = np.argmin(sel_ranges)
            occu_map_polar[:sel_range_idx[closest_range_idx], i] = 0.02
    return occu_map_polar


def popoulate_free_space_polar_grid_v2(
    polar_grid_prop, 
    occu_map_polar, 
    rangecoord, 
    azimuthcoord):
    """ polulate the free space probabilities in polar grid given the measurements in polar coordinate frame
    Here the polar grid has variable sized grid resolution and the grid cells are overlapping
    Input: polar_grid_prop - polar grid object which contains the properties of the polar grid
         : occu_map_polar - polar occupancy grid
         : rangecoord - measurement range 
         : azimuthcoord - measurement azimuth angle
    Output : occu_map_polar - polar grid containing the free space probabilities
    """
    rangeids, _ = polar_grid_prop.compute_ra_ids_from_ra_coordinates(rangecoord, azimuthcoord)
    for i in range(polar_grid_prop.num_cells_azimuth):
        flag = np.logical_and( 
            azimuthcoord >= polar_grid_prop.theta_lower[i], 
            azimuthcoord <= polar_grid_prop.theta_upper[i])
        cond = np.sum(flag.astype(np.int32))
        if cond == 0: occu_map_polar[:, i] = 0.02
        elif cond > 0:
            sel_ranges = rangecoord[flag]
            sel_range_idx = rangeids[flag]
            closest_range_idx = sel_range_idx[np.argmin(sel_ranges)]
            occu_map_polar[:closest_range_idx, i] = 0.02
    return occu_map_polar


def popoulate_free_space_polar_grid_v3(
    polar_grid_prop, 
    occu_map_polar, 
    rangecoord, 
    azimuthcoord,
    num_beams):
    """ polulate the free space probabilities in polar grid given the measurements in polar coordinate frame
    Here the polar grid has variable sized grid resolution and the grid cells are overlapping and
    while populating the free space a beam of azimuth rays are considered rather than a single ray
    Input: polar_grid_prop - polar grid object which contains the properties of the polar grid
         : occu_map_polar - polar occupancy grid
         : rangecoord - measurement range 
         : azimuthcoord - measurement azimuth angle
         : num_beams - size of the ray bundle 
    Output : occu_map_polar - polar grid containing the free space probabilities
    """
    rangeids, _ = polar_grid_prop.compute_ra_ids_from_ra_coordinates(rangecoord, azimuthcoord)
    for i in range(polar_grid_prop.num_cells_azimuth):
        flag = np.logical_and( 
            azimuthcoord >= polar_grid_prop.theta_lower[i], 
            azimuthcoord <= polar_grid_prop.theta_upper[i])
        cond = np.sum(flag.astype(np.int32))
        if cond == 0: occu_map_polar[:, i] = 0.02
        elif cond > 0:
            sel_ranges = rangecoord[flag]
            sel_range_idx = rangeids[flag]
            closest_range_idx = sel_range_idx[np.argmin(sel_ranges)]
            occu_map_polar[:closest_range_idx, i] = 0.02
            for j in range(1, num_beams):
                i_next = np.clip(i+j, 0, polar_grid_prop.num_cells_azimuth-1)
                i_prev = np.clip(i-j, 0, polar_grid_prop.num_cells_azimuth-1)
                occu_map_polar[:closest_range_idx, i_next] = 0.02
                occu_map_polar[:closest_range_idx, i_prev] = 0.02
    return occu_map_polar


def set_free_space_info(
    occ_polar, 
    occ_cart, 
    cart_grid_prop, 
    cart_cell_ids, 
    rids, 
    aids):
    """ set the free space info from polar to cartesian
    inputs: occ_polar - grid image in polar 
          : occ_cart - grid image in cartesian
          : cart_grid_prop - cartesian grid properties
          : cart_cell_ids - mapped cartesian cell ids
          : rids, aids - corrosponding ids to polar grid 
    """
    xids, yids = cart_grid_prop.compute_xy_ids_from_scalar_ids(cart_cell_ids)
    occ_cart[xids, yids] = occ_polar[rids, aids]
    return occ_cart


def compute_coordinates_for_free_space(
    occ_cart, 
    cart_grid_prop,
    free_threshold):
    """ compute free space coordinates by thresholding the occupancy probability
    Inputs: occ_cart - grid image in cartesian
          : cart_grid_obj - cartesian grid properties 
          : free_threshold - threshold parameter for free space determination
    Outputs: xcoord, ycoord - x and y coordinates of the free space grid cells
    """
    xcoord = cart_grid_prop.x_coord
    ycoord = cart_grid_prop.y_coord
    ycoord, xcoord = np.meshgrid(ycoord, xcoord)
    xycoord = np.reshape(np.stack([xcoord, ycoord], axis=-1), (-1, 2))

    free_space_flag = occ_cart < free_threshold
    free_space_flag = np.reshape(free_space_flag, -1)
    xycoord_free = xycoord[free_space_flag]
    return xycoord_free[:, 0], xycoord_free[:, 1]


def set_occupied_space_info(
    occ_cart, 
    cart_grid_prop, 
    log_odds_grid_prop, 
    cellids, 
    log_odds):
    """ set the occupied space info in cartesian
    Inputs: occ_cart - grid image in cartesian
          : cart_grid_obj - cartesian grid properties
          : log_odds_grid_obj - meas log-odds grid properties 
          : cellids - cells ids of the meas
          : log_odds - log-odds values of the meas
    """
    xcoord, ycoord = log_odds_grid_prop.compute_xy_coordinates_from_scalar_ids(cellids)
    condition = select_points_within_grid_boundary(
        xcoord, ycoord, 
        cart_grid_prop.min_x, cart_grid_prop.max_x,
        cart_grid_prop.min_y, cart_grid_prop.max_y)

    xcoord_occupied = xcoord[condition]
    ycoord_occupied = ycoord[condition]
    log_odds = log_odds[condition]

    xids, yids = cart_grid_prop.compute_xy_ids_from_xy_coordinates(xcoord_occupied, ycoord_occupied)
    occ_cart[xids, yids] = compute_prob_from_log_odds(log_odds)
    return occ_cart, xcoord_occupied, ycoord_occupied, log_odds


def init_grid_maps(
    cart_grid_prop, 
    polar_grid_prop):
    """ Initialize the grid maps
    Input: cart_grid_obj - cartesian grid object which contains the properties of the cartesian grid
         : polar_grid_obj - polar grid object which contains the properties of the polar grid
    Output: occu_map_polar - initialized polar grid
          : occu_map_cart - initialized cartesian grid
    """
    occu_map_polar = 0.5 + np.zeros((polar_grid_prop.num_cells_range, polar_grid_prop.num_cells_azimuth), dtype=np.float32)
    occu_map_cart = 0.5 + np.zeros((cart_grid_prop.num_cells_x, cart_grid_prop.num_cells_y), dtype=np.float32)
    return occu_map_polar, occu_map_cart


def occupancy_grid_mapping_ray_casting_in_polar(
    free_space_threshold,
    log_odds_grid_prop,
    occu_cart_grid_prop,
    occu_polar_grid_prop,
    log_odds,
    cellids):
    """ Occupancy grid mapping by ray casting in the polar grid using selected (high confident) measurements
    Inputs: log_odds_threshold - 
          : free_space_threshold - 
          : log_odds_grid_prop - 
          : occu_cart_grid_prop -
          : occu_polar_grid_prop -
          : log_odds -
          : cellids -
    Outputs: occu_map_cart -
           : log_odds_obs -
           : xcoord_free -
           : ycoord_free -
           : xcoord_occupied -
           : ycoord_occupied -
    """
    # initialize
    occu_map_polar, occu_map_cart = init_grid_maps(occu_cart_grid_prop, occu_polar_grid_prop)

    # compute cell coordinates, convert cell from cartesian to polar, choose meas that are within the polar grid
    rangecoord, azimuthcoord = select_points_within_polar_roi(log_odds_grid_prop, occu_polar_grid_prop, cellids)

    # update free sapce in polar grid
    occu_map_polar = popoulate_free_space_polar_grid_v2(occu_polar_grid_prop, occu_map_polar, rangecoord, azimuthcoord)

    # map cart grid to polar grid
    cart_cell_ids, rids, aids = map_cartesian_grid_ids_to_polar_grid(occu_cart_grid_prop, occu_polar_grid_prop)
    
    # update free sapce in cartesian grid
    occu_map_cart = set_free_space_info(occu_map_polar, occu_map_cart, occu_cart_grid_prop, cart_cell_ids, rids, aids)
    xcoord_free, ycoord_free = compute_coordinates_for_free_space(occu_map_cart, occu_cart_grid_prop, free_space_threshold)

    # update occupied sapce in cartesian grid
    occu_map_cart, xcoord_occupied, ycoord_occupied, log_odds_obs \
        = set_occupied_space_info(occu_map_cart, occu_cart_grid_prop, log_odds_grid_prop, cellids, log_odds)

    return (
        occu_map_cart, log_odds_obs,
        xcoord_free, ycoord_free, 
        xcoord_occupied, ycoord_occupied )


def road_boundary_point_extraction(
    polar_grid_prop,
    xcoord, 
    ycoord):

    # init
    boundary_range = []
    boundary_azimuth = []

    # convert coordinates from cartesian to polar
    rangecoord, azimuthcoord = convert_grid_cart_to_polar(xcoord, ycoord)

    # choose meas that are within the polar grid
    condition \
        = select_points_within_grid_boundary(
            rangecoord, azimuthcoord, 
            polar_grid_prop.min_range, polar_grid_prop.max_range,
            polar_grid_prop.min_azimuth, polar_grid_prop.max_azimuth)

    rangecoord = rangecoord[condition]
    azimuthcoord = azimuthcoord[condition]

    # for each azimuth beam extract the smallest range
    for i in range(polar_grid_prop.num_cells_azimuth):

        flag = np.logical_and( 
            azimuthcoord >= polar_grid_prop.theta_lower[i], 
            azimuthcoord <= polar_grid_prop.theta_upper[i])

        if np.sum(flag.astype(np.int32)) > 0:

            sel_ranges = rangecoord[flag]
            sel_azimuths = azimuthcoord[flag]
            idx = np.argmin(sel_ranges)

            boundary_range.append(sel_ranges[idx])
            boundary_azimuth.append(sel_azimuths[idx])

    boundary_range = np.array(boundary_range)
    boundary_azimuth = np.array(boundary_azimuth)

    # convert it back to cartesian coordinates
    boundary_xcoord, boundary_ycoord = convert_grid_polar_to_cartesian(boundary_range, boundary_azimuth)

    return (
        boundary_xcoord, 
        boundary_ycoord )


