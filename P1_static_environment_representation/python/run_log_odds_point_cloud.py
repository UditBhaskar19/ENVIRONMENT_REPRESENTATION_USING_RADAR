# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Animation script for visualizing log-odds point cloud
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def set_plot_properties(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return ax

# ===================================================================================================================

class animate(object):

    def __init__(self, ax):

        self.scene = scene
        self.data_dir = data_dir
        self.data_odom = data_odom
        self.num_samples = num_samples
        self.frame_offset = frame_offset

        self.T_prev = np.eye(3, dtype=np.float32)
        self.trigger = np.ones((lib_const.num_radars, ), dtype=np.bool8)
        self.tracked_cell_states = cell_states(config.max_num_cells, lib_const.num_radars)
        self.logodds_eps = 4.0

        ax.scatter(0, 0, s=150, marker='*', color='black', label='ego vehicle wheel base centre')
        self.stationary_meas = ax.scatter([],[], s=0.5, color='red', marker='.', label='dense radar meas')
        ax.legend(loc='upper right')
        self.ax = ax
        

    def __call__(self, t):

        print(t)

        # compute log-odds at each t
        logodds, cellids, xcoord, ycoord, \
        self.tracked_cell_states, \
        self.T_prev, \
        self.trigger, \
        timestamp_rad = meas_log_odds_map_step(
                self.tracked_cell_states,
                self.trigger,
                grid_prop, 
                fov_coverage.fov_coverage_flag,
                self.data_odom,
                self.frame_offset + t,
                self.T_prev,
                self.logodds_eps)

        # compute the cell coordinates and the probabilities from log-odds
        prob = 2 * compute_prob_from_log_odds(logodds) - 1

        # plot the cloud
        meas = np.stack([xcoord, ycoord], axis=-1)
        self.stationary_meas.set_offsets(meas)
        # self.stationary_meas.set_alpha(prob)
        self.ax.set_title('Scene ' + str(self.scene) + ': Log-Odds Point Cloud at time ' + f'{timestamp_rad:.2f}' + ' sec', fontsize=14)




fig, ax = plt.subplots()
ax = set_plot_properties(ax, x_min=-55, x_max=105, y_min=-55, y_max=55)
integrated_scans = animate(ax)
anim = FuncAnimation(fig, integrated_scans, frames=np.arange(n), interval=0.001, repeat=False)
plt.show()



