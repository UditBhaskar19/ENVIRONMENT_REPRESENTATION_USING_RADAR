# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library for Grid Cell State Update
# =====================================================================================================================

import lib_const, config
import numpy as np
from lib_grid import (
    select_points_within_grid_boundary, 
    convert_meas_polar_to_cartesian
)
from lib_meas_selection import (
    coordinate_transform_px_py,
    select_stationary_measurements
)
from lib_log_odds import (
    compute_meas_log_likelihood,
    inflate_prob_and_compute_log_odds
)
from lib_scan_accum import (
    sync_radar_with_egomotion, 
    construct_SE2_group_element,
    ego_compensate_prev_meas_vehicle_frame
)

def cell_states_prediction(
    grid_properties,
    logodds, 
    xcoord, 
    ycoord,
    T_curr, 
    T_prev):

    xcoord, ycoord = ego_compensate_prev_meas_vehicle_frame(xcoord, ycoord, T_curr, T_prev)
    condition \
        = select_points_within_grid_boundary(
            xcoord, ycoord, 
            grid_properties.min_x, grid_properties.max_x,
            grid_properties.min_y, grid_properties.max_y)

    xcoord = xcoord[condition]
    ycoord = ycoord[condition]
    logodds = logodds[condition]
    cellids = grid_properties.compute_scalar_ids_from_xy_coordinates(xcoord, ycoord)

    return ( logodds, cellids, xcoord, ycoord )

# =================================================================================================================

def determine_association_type(
    num_cells, 
    cell_within_fov,
    cellids_pred,
    cellids_meas):

    # cell type computation
    flag_pred = np.zeros((num_cells, ), dtype=np.bool8)
    flag_meas = np.zeros((num_cells, ), dtype=np.bool8)

    flag_pred[cellids_pred] = True
    flag_meas[cellids_meas] = True

    gated_cells = np.logical_and(flag_pred, flag_meas)
    new_cells = np.logical_and( flag_meas, np.logical_not( flag_pred ) )

    ungated_cells = np.logical_and(flag_pred, np.logical_not(flag_meas))
    ungated_cells_within_fov = np.logical_and(cell_within_fov, ungated_cells)
    ungated_cells_outside_fov = np.logical_and(np.logical_not(cell_within_fov), ungated_cells)

    return (
        flag_pred, 
        flag_meas,
        gated_cells, 
        new_cells,
        ungated_cells,
        ungated_cells_within_fov, 
        ungated_cells_outside_fov )

# =================================================================================================================

def update_logodds(
    num_cells,
    cellids_pred, 
    cellids_meas,
    logodds_pred, 
    logodds_meas,
    new_cells, 
    gated_cells,
    ungated_cells_within_fov,
    ungated_cells_outside_fov):

    # log odds prediction and update
    log_odds_pred_vals = np.zeros((num_cells, ), dtype=np.float32)
    log_odds_meas_vals = np.zeros((num_cells, ), dtype=np.float32)
    log_odds_upd = np.zeros((num_cells, ), dtype=np.float32)

    log_odds_pred_vals[cellids_pred] = logodds_pred
    log_odds_meas_vals[cellids_meas] = logodds_meas

    log_odds_upd[new_cells] = init_new_log_odds(log_odds_meas_vals[new_cells])
    log_odds_upd[gated_cells] = update_log_odds(log_odds_meas_vals[gated_cells], log_odds_pred_vals[gated_cells])
    log_odds_upd[ungated_cells_within_fov] = update_log_odds_ungated_within_fov(log_odds_pred_vals[ungated_cells_within_fov])
    log_odds_upd[ungated_cells_outside_fov] = update_log_odds_ungated_outside_fov(log_odds_pred_vals[ungated_cells_outside_fov])

    return log_odds_upd

# =================================================================================================================

def update_cell_coordinates(
    num_cells, 
    cellids_pred, 
    cellids_meas,
    xcoord_pred,
    xcoord_meas,
    ycoord_pred, 
    ycoord_meas,
    new_cells, 
    gated_cells, 
    ungated_cells ):

    cell_coord_upd = np.zeros((num_cells, 2), dtype=np.float32)
    cell_coord_pred_vals = np.zeros((num_cells, 2), dtype=np.float32)
    cell_coord_meas_vals = np.zeros((num_cells, 2), dtype=np.float32)

    cell_coord_pred_vals[cellids_pred] = np.stack([xcoord_pred, ycoord_pred], axis=-1)
    cell_coord_meas_vals[cellids_meas] = np.stack([xcoord_meas, ycoord_meas], axis=-1)

    cell_coord_upd[ungated_cells] = cell_coord_pred_vals[ungated_cells]
    cell_coord_upd[new_cells] = cell_coord_meas_vals[new_cells]
    cell_coord_upd[gated_cells] = cell_coord_meas_vals[gated_cells]

    return ( 
        cell_coord_upd[:, 0], 
        cell_coord_upd[:, 1] )

# =================================================================================================================

def sort_logodds_and_compute_cellids(
    num_cells,
    flag_pred, 
    flag_meas,
    logodds_upd,
    xcoord_upd,
    ycoord_upd):

    flag = np.logical_or(flag_pred, flag_meas)
    cellids_upd = np.arange(num_cells, dtype=np.uint32)

    cellids_upd = cellids_upd[flag]
    logodds_upd = logodds_upd[flag]
    xcoord_upd = xcoord_upd[flag]
    ycoord_upd = ycoord_upd[flag]
    idx = np.argsort(logodds_upd)

    return (
        logodds_upd[idx],
        cellids_upd[idx],
        xcoord_upd[idx],
        ycoord_upd[idx] )

# =================================================================================================================

def cell_states_update(
    fov_coverage_flag, 
    logodds_pred, 
    cellids_pred,
    xcoord_pred, 
    ycoord_pred,
    logodds_meas, 
    cellids_meas,
    xcoord_meas, 
    ycoord_meas):

    num_cells = fov_coverage_flag.shape[0]

    # cell type computation
    flag_pred, flag_meas, \
    gated_cells, new_cells, \
    ungated_cells, \
    ungated_cells_within_fov, \
    ungated_cells_outside_fov \
        = determine_association_type(
            num_cells, fov_coverage_flag,
            cellids_pred, cellids_meas)

    # update log odds
    logodds_upd \
        = update_logodds(
            num_cells, cellids_pred, cellids_meas,
            logodds_pred, logodds_meas,
            new_cells, gated_cells,
            ungated_cells_within_fov,
            ungated_cells_outside_fov)

    # update cell coordinates
    xcoord_upd, ycoord_upd \
        = update_cell_coordinates(
            num_cells, cellids_pred, cellids_meas,
            xcoord_pred, xcoord_meas,
            ycoord_pred, ycoord_meas,
            new_cells, gated_cells, ungated_cells
        )

    # sort log ids and compute cell ids
    logodds_upd, cellids_upd, \
    xcoord_upd, ycoord_upd \
        = sort_logodds_and_compute_cellids(
            num_cells, flag_pred, flag_meas,
            logodds_upd, xcoord_upd, ycoord_upd)

    return (
        logodds_upd, cellids_upd,
        xcoord_upd, ycoord_upd )

# =================================================================================================================

def init_new_log_odds(log_odds_meas):
    log_odds_meas = 0.8 * log_odds_meas
    return log_odds_meas
    
def update_log_odds(log_odds_meas, log_odds_pred):
    log_odds_pred = 0.99 * log_odds_pred + log_odds_meas
    return log_odds_pred

def update_log_odds_ungated_within_fov(log_odds_pred):
    log_odds_pred = 0.85 * log_odds_pred
    return log_odds_pred

def update_log_odds_ungated_outside_fov(log_odds_pred):
    return log_odds_pred

# =================================================================================================================

def log_odds_filter_step(
    rad_meas,
    tx, ty, yaw,
    odom_vx, 
    odom_yawrate, 
    x_loc, y_loc, yaw_loc,
    timestamp_rad, 
    timestamp_odom,
    grid_properties,
    trigger,
    fov_coverage_flag,
    T_prev,
    cell_states,
    radar_id):

    # identify stationary measurements
    meas_stationary = select_stationary_measurements(
        rad_meas, (tx, ty, yaw), 
        odom_vx, odom_yawrate)

    # polar to cartesian conversion
    meas_stationary_x, \
    meas_stationary_y \
        = convert_meas_polar_to_cartesian(
            meas_stationary[:, lib_const.rad_meas_attr['range']], 
            meas_stationary[:, lib_const.rad_meas_attr['azimuth']])

    # coordinate transformation from sensor frame to vehicle frame
    meas_stationary_x, \
    meas_stationary_y \
        = coordinate_transform_px_py(
            meas_stationary_x, meas_stationary_y, 
            tx, ty, yaw)

    # sync radar measurements with odom
    meas_stationary_x, \
    meas_stationary_y  \
        = sync_radar_with_egomotion(
            meas_stationary_x, meas_stationary_y, timestamp_rad, 
            odom_vx, odom_yawrate, timestamp_odom)

    # compute log odds
    prob_meas, cellids_meas, \
    xcoord_meas, ycoord_meas \
        = compute_meas_log_likelihood(
            meas_stationary_x,
            meas_stationary_y, 
            config.sampling_variance,
            config.num_particles,
            grid_properties)

    # scale and add offset so that the probabilities of samples are between [0.5, 1]    
    logodds_meas = inflate_prob_and_compute_log_odds(prob_meas)

    # create the current pose
    T_curr = construct_SE2_group_element(x_loc, y_loc, yaw_loc)

    # trigger == True indicates if the measurements are received for the first time from radar i
    if trigger[radar_id]:
        logodds_upd = logodds_meas
        cellids_upd = cellids_meas
        xcoord_upd = xcoord_meas
        ycoord_upd = ycoord_meas
        trigger[radar_id] = False

    else:
        logodds_pred, cellids_pred, \
        xcoord_pred, ycoord_pred \
            = cell_states_prediction(
                grid_properties, 
                cell_states.get_logodds_sensor_i(radar_id),
                cell_states.get_px_sensor_i(radar_id), 
                cell_states.get_py_sensor_i(radar_id),
                T_curr, T_prev)

        logodds_upd, cellids_upd, \
        xcoord_upd, ycoord_upd \
            = cell_states_update(
                fov_coverage_flag[radar_id], 
                logodds_pred, cellids_pred,
                xcoord_pred, ycoord_pred,
                logodds_meas, cellids_meas,
                xcoord_meas, ycoord_meas)

    # only select top-k log-odds
    N = logodds_upd.shape[0]
    L = 0
    if N > config.max_num_cells:
        L = N - config.max_num_cells
        N = config.max_num_cells

    # update the information
    cell_states.set_num_tracked_cells_sensor_i(radar_id, N)
    cell_states.set_logodds_sensor_i(radar_id, logodds_upd[L:])
    cell_states.set_cellids_sensor_i(radar_id, cellids_upd[L:])
    cell_states.set_px_sensor_i(radar_id, xcoord_upd[L:])
    cell_states.set_py_sensor_i(radar_id, ycoord_upd[L:])

    return (
        trigger,
        cell_states, 
        T_curr
    )

# =================================================================================================================

def merge_log_odds_multiple_sensors(
    cell_states,
    grid_properties,
    trigger):

    cellids = np.arange(grid_properties.num_cells, dtype=np.uint32)
    flag = np.zeros((lib_const.num_radars, grid_properties.num_cells), dtype=np.bool8)
    logodds = np.zeros((lib_const.num_radars, grid_properties.num_cells), dtype=np.float32)
    xcoord = np.zeros((lib_const.num_radars, grid_properties.num_cells), dtype=np.float32)
    ycoord = np.zeros((lib_const.num_radars, grid_properties.num_cells), dtype=np.float32)
    
    for i in range(lib_const.num_radars):
        if trigger[i] == False:
            logodds[i, cell_states.get_cellids_sensor_i(i)] = cell_states.get_logodds_sensor_i(i)
            xcoord[i, cell_states.get_cellids_sensor_i(i)] = cell_states.get_px_sensor_i(i)  
            ycoord[i, cell_states.get_cellids_sensor_i(i)] = cell_states.get_py_sensor_i(i) 
            flag[i, cell_states.get_cellids_sensor_i(i)] = True

    # sum of the log-odds and the mean of the positions
    logodds = np.sum(logodds, axis=0)
    n = np.sum(flag, axis=0)
    n = np.where(n!=0, n, lib_const.eps)
    xcoord = np.sum(xcoord * flag.astype(np.float32), axis=0) / n
    ycoord = np.sum(ycoord * flag.astype(np.float32), axis=0) / n

    or_bool_cond = np.zeros((grid_properties.num_cells, ), dtype=np.bool8)
    for i in range(lib_const.num_radars):
        or_bool_cond = np.logical_or(or_bool_cond, flag[i])

    # set the fused cell state info
    cell_states.set_num_fused_tracked_cells(np.sum(or_bool_cond).astype(np.uint32))
    cell_states.set_fused_logodds(logodds[or_bool_cond])
    cell_states.set_fused_cellids(cellids[or_bool_cond])
    cell_states.set_fused_px(xcoord[or_bool_cond])
    cell_states.set_fused_py(ycoord[or_bool_cond])
    return cell_states

# =================================================================================================================

def ego_compensate_cells_other_sensors(
    grid_properties,
    cell_states,
    radar_id,
    T_curr,
    T_prev,
    trigger):

    for i in range(lib_const.num_radars):
        # condition = ( i != radar_id and trigger[i] == False )
        if i != radar_id and trigger[i] == False :
            logodds, cellids, xcoord, ycoord \
                = cell_states_prediction(
                    grid_properties,
                    cell_states.get_logodds_sensor_i(i),
                    cell_states.get_px_sensor_i(i), 
                    cell_states.get_py_sensor_i(i) ,
                    T_curr, T_prev)

            cell_states.set_num_tracked_cells_sensor_i(i, cellids.shape[0])
            cell_states.set_logodds_sensor_i(i, logodds)
            cell_states.set_cellids_sensor_i(i, cellids)
            cell_states.set_px_sensor_i(i, xcoord)
            cell_states.set_py_sensor_i(i, ycoord)

    return cell_states

# =================================================================================================================

def filter_log_odds_by_thresholding(cell_states, threshold):
    logodds = cell_states.get_fused_logodds()
    cellids = cell_states.get_fused_cellids()
    xcoord = cell_states.get_fused_px()
    ycoord = cell_states.get_fused_py()
    flag = logodds > threshold
    return ( 
        logodds[flag], 
        cellids[flag],
        xcoord[flag],
        ycoord[flag] )


        