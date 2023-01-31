# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Animation script for visualizing log-odds image 
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from lib_datastruct import cell_states
from lib_read_data import (
    parse_csv_file, 
    count_csv_files,
    extract_sensor_sync_data,
    extract_radar_data
)
from lib_grid import grid_properties, FOV_grid_coverage
from lib_log_odds import compute_prob_from_log_odds
from lib_log_odds_kalman_filter import (
    log_odds_filter_step,
    ego_compensate_cells_other_sensors,
    merge_log_odds_multiple_sensors,
    filter_log_odds_by_thresholding
)

scene = '105'
data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
_, data_odom = parse_csv_file(os.path.join(data_dir, 'timestamp_and_odom.csv'))
num_samples = count_csv_files(data_dir) - 1
frame_offset = 0
n = num_samples

grid_prop = grid_properties(
        min_x = -50.0, 
        max_x =  100.0, 
        min_y = -50.0, 
        max_y =  50.0, 
        dx = 0.2, 
        dy = 0.2)

fov_coverage = FOV_grid_coverage(grid_prop)
xy_grid_coord = np.stack([fov_coverage.x_coord, fov_coverage.y_coord], axis=-1)

# ===================================================================================================================

def meas_log_odds_map_step(
    cell_states,
    trigger,
    grid_prop, 
    fov_coverage_flag,
    odom_data, 
    frameid,
    T_prev,
    log_odds_eps):

    # extract time stamps and odom
    odom_vx, _, odom_yawrate, x_loc, y_loc, yaw_loc, \
    radar_id, timestamp_rad, timestamp_odom = extract_sensor_sync_data(odom_data, frameid)

    # extract the radar frame data
    rad_meas = extract_radar_data(data_dir, frameid)

    # extract radar mount param
    mount_param = lib_const.radars_mount[radar_id]
    tx = mount_param[0]
    ty = mount_param[1]
    yaw = mount_param[2]

    # perform log odds filter step for radar i
    trigger, \
    cell_states, \
    T_curr = log_odds_filter_step(
        rad_meas, tx, ty, yaw,
        odom_vx, odom_yawrate, 
        x_loc, y_loc, yaw_loc,
        timestamp_rad, timestamp_odom, 
        grid_prop, trigger, fov_coverage_flag,
        T_prev, cell_states,
        radar_id )

    # synchronize
    cell_states \
        = ego_compensate_cells_other_sensors(
            grid_prop, cell_states, 
            radar_id, T_curr, T_prev, 
            trigger)

    # merge log-odds from multiple sensors
    cell_states =  merge_log_odds_multiple_sensors(cell_states, grid_prop, trigger)

    # filter out log-odds if below a threshold
    logodds, cellids, xcoord, ycoord = filter_log_odds_by_thresholding(cell_states, log_odds_eps)

    return (
        logodds, cellids, xcoord, ycoord,
        cell_states, T_curr, trigger, timestamp_rad )

# ===================================================================================================================

def main(scene):

    T_prev = np.eye(3, dtype=np.float32)
    trigger = np.ones((lib_const.num_radars, ), dtype=np.bool8)
    tracked_cell_states = cell_states(config.max_num_cells, lib_const.num_radars)
    logodds_eps = 0.0

    _, ax = plt.subplots()
    
    for t in range(n):

        # compute log-odds at each t
        logodds, cellids, xcoord, ycoord, \
        tracked_cell_states, \
        T_prev, \
        trigger, \
        timestamp_rad = meas_log_odds_map_step(
                tracked_cell_states,
                trigger,
                grid_prop, 
                fov_coverage.fov_coverage_flag,
                data_odom,
                frame_offset + t,
                T_prev,
                logodds_eps)

        # for visualization
        cellxid, cellyid = grid_prop.compute_xy_ids_from_scalar_ids(cellids)
        prob_pred_img = np.zeros((grid_prop.num_cells_x, grid_prop.num_cells_y), dtype=np.float32)
        prob_pred_img[cellxid, cellyid] = 2 * compute_prob_from_log_odds(logodds) - 1

        ax.clear()
        ax.imshow(np.flip(prob_pred_img.transpose(), axis=0))
        ax.set_title('Scene ' + str(scene) + ': Log-Odds Map at time ' + f'{timestamp_rad:.2f}' + ' sec', fontsize=14)
        plt.pause(0.01)

        plt.show(block=False)
        if not plt.get_fignums():
            exit()

# ===================================================================================================================

if __name__ == '__main__':
    scene = '105'
    main(scene)