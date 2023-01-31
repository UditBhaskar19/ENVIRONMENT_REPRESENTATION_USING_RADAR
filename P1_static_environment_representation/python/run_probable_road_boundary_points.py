# =============================================================================================================================
# Author Name : Udit Bhaskar
# description : Animation script for visualizing a basic version road boundary point determination by ray-casting in polar grid 
# ==============================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib_read_data import (
    parse_csv_file, 
    count_csv_files,
    extract_sensor_sync_data,
    extract_radar_data
)
from lib_grid import (
    grid_properties, 
    FOV_grid_coverage,
    polar_grid_properties
)
from lib_log_odds_kalman_filter import (
    log_odds_filter_step,
    ego_compensate_cells_other_sensors,
    merge_log_odds_multiple_sensors,
    filter_log_odds_by_thresholding
)
from lib_datastruct import cell_states
from lib_use_cases import road_boundary_point_extraction

scene = '105'
data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
_, data_odom = parse_csv_file(os.path.join(data_dir, 'timestamp_and_odom.csv'))
num_samples = count_csv_files(data_dir) - 1
frame_offset = 0
n = num_samples

grid_prop = grid_properties(
        min_x = -50.0, 
        max_x =  100.0, 
        min_y = -40.0, 
        max_y =  40.0, 
        dx = 0.2, 
        dy = 0.2)
fov_coverage = FOV_grid_coverage(grid_prop)

polar_grid_prop = polar_grid_properties( 
        min_range = 0.0, 
        max_range = 100.0, 
        min_azimuth = -180, 
        max_azimuth = 180, 
        range_res = 0.5, 
        azimuth_res = 0.15,
        theta_res_min = 1.2,
        theta_res_max = 10.0 )

# ===================================================================================================================

def road_boundary_est_step(
    cell_states,
    trigger,
    grid_prop,
    polar_grid_prop, 
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
    _, _, meas_xcoord, meas_ycoord = filter_log_odds_by_thresholding(cell_states, log_odds_eps)

    # extract probable road boundary
    boundary_xcoord, boundary_ycoord = road_boundary_point_extraction( polar_grid_prop, meas_xcoord, meas_ycoord)

    return (
        cell_states, 
        T_curr, 
        trigger, 
        timestamp_rad,
        boundary_xcoord, 
        boundary_ycoord,
        meas_xcoord, 
        meas_ycoord )

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
        self.logodds_eps = 15.7

        self.occu_space = ax.scatter([],[], s=0.5, color='red', marker='.')
        self.boundary_points = ax.scatter([],[], s=30, color='black', marker='x', label='probable road boundary point')
        ax.scatter(0, 0, s=150, marker='*', color='green', label='ego vehicle wheel base centre')
        ax.legend(loc='upper right')
        self.ax = ax
        

    def __call__(self, t):

        print(t)

        # compute free sapce at time t
        self.tracked_cell_states, \
        self.T_prev, \
        self.trigger, \
        timestamp_rad, \
        boundary_xcoord, \
        boundary_ycoord, \
        meas_xcoord, \
        meas_ycoord = road_boundary_est_step(
                self.tracked_cell_states,
                self.trigger,
                grid_prop,
                polar_grid_prop, 
                fov_coverage.fov_coverage_flag,
                self.data_odom, 
                self.frame_offset + t,
                self.T_prev,
                self.logodds_eps )

        # plot the cloud
        self.occu_space.set_offsets(np.stack([meas_xcoord, meas_ycoord], axis=-1))
        self.boundary_points.set_offsets(np.stack([boundary_xcoord, boundary_ycoord], axis=-1))
        self.ax.set_title('Scene ' + str(self.scene) + ': Probable road border points at time ' + f'{timestamp_rad:.2f}' + ' sec', fontsize=14)


fig, ax = plt.subplots()
ax = set_plot_properties(ax, x_min=-55, x_max=105, y_min=-45, y_max=45)
integrated_scans = animate(ax)
anim = FuncAnimation(fig, integrated_scans, frames=np.arange(n), interval=1, repeat=False)
plt.show()



