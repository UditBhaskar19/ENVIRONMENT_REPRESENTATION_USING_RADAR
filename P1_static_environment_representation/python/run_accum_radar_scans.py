# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : ANIMATION script for visualizing radar scan accumulations
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib_datastruct import meas_hist_buffer
from lib_grid import (
    convert_meas_polar_to_cartesian, 
    grid_properties, 
    FOV_grid_coverage
)
from lib_read_data import (
    parse_csv_file, 
    count_csv_files, 
    extract_sensor_sync_data, 
    extract_radar_data
)
from lib_meas_selection import (
    select_stationary_measurements,
    coordinate_transform_px_py
)
from lib_scan_accum import (
    inverse_SE2,
    coordinate_transform_meas,
    sync_radar_with_egomotion,
    construct_SE2_group_element,
    ego_compensate_prev_meas_vehicle_frame
)


scene = '105'
data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
_, data_odom = parse_csv_file(os.path.join(data_dir, 'timestamp_and_odom.csv'))
num_samples = count_csv_files(data_dir) - 1
frame_offset = 0
n = num_samples


GridCartesian = grid_properties(
        min_x = -90.0, 
        max_x = 120.0, 
        min_y = -120.0, 
        max_y = 120.0, 
        dx = 0.5, 
        dy = 0.5
    )

fov_coverage = FOV_grid_coverage(GridCartesian)
xy_grid_coord = np.stack([fov_coverage.x_coord, fov_coverage.y_coord], axis=-1)

# ===================================================================================================================

def meas_integration_step(
    data_dir, 
    odom_data, 
    frameid,
    trigger,
    meas_hist,
    T_prev
    ):

    # extract time stamps and odom
    odom_vx, _, odom_yawrate, x_loc, y_loc, yaw_loc, \
    radar_id, timestamp_rad, timestamp_odom = extract_sensor_sync_data(odom_data, frameid)

    # extract the radar data and radar mount info
    rad_meas = extract_radar_data(data_dir, frameid)
    mount_param = lib_const.radars_mount[radar_id]

    # extract stationary measurements
    meas_stationary = select_stationary_measurements(
        rad_meas, (mount_param[0], mount_param[1], mount_param[2]), 
        odom_vx, odom_yawrate)

    # polar to cartesian meas
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
            mount_param[0], mount_param[1], mount_param[2])

    # sync radar measurements with odom
    meas_stationary_x, \
    meas_stationary_y  \
        = sync_radar_with_egomotion(
            meas_stationary_x, meas_stationary_y, timestamp_rad,
            odom_vx, odom_yawrate, timestamp_odom)

    # integrate frmes
    if trigger: 
        T_prev = construct_SE2_group_element(x_loc, y_loc, yaw_loc)
        num_meas = meas_stationary_x.shape[0]
        meas_hist.update_buffer(
            meas_stationary_x, meas_stationary_y, 
            num_meas, radar_id, timestamp_rad, 
            x_loc, y_loc, yaw_loc)

    else:
        T_curr = construct_SE2_group_element(x_loc, y_loc, yaw_loc)
        meas_hist.meas_xcoord, \
        meas_hist.meas_ycoord \
            = ego_compensate_prev_meas_vehicle_frame(
                meas_hist.meas_xcoord, meas_hist.meas_ycoord,
                T_curr, T_prev)
        T_prev = T_curr
        
        num_meas = meas_stationary_x.shape[0]
        meas_hist.update_buffer(
            meas_stationary_x, meas_stationary_y, 
            num_meas, radar_id, timestamp_rad, 
            x_loc, y_loc, yaw_loc)

    print('frameid: ', frameid, '  num_meas: ', meas_hist.meas_xcoord.shape[0], '     time: ', f'{timestamp_rad:.3f}')
    return meas_hist, T_prev, timestamp_rad

# ===================================================================================================================

def set_plot_properties(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # ax.set_title('Radar scan visualization w.r.t the ego vehicle frame')
    return ax

# ===================================================================================================================

class animate(object):

    def __init__(self, ax):

        self.scene = scene
        self.data_dir = data_dir
        self.data_odom = data_odom
        self.num_samples = num_samples
        self.frame_offset = frame_offset

        self.T_prev = 0
        self.trigger = True
        self.buffer_size = 150
        self.meas_hist = meas_hist_buffer(self.buffer_size)

        # radar fov coordinates
        fov_coverage_flag = fov_coverage.fov_coverage_flag
        fov_coverage_coord_flag = np.zeros((xy_grid_coord.shape[0],), dtype=np.bool8)
        for i in range(lib_const.num_radars):
            fov_coverage_coord_flag = np.logical_or(fov_coverage_coord_flag, fov_coverage_flag[i])
        fov_coverage_xy_grid_coord = xy_grid_coord[fov_coverage_coord_flag]

        self.radar_fov \
            = ax.scatter(
                fov_coverage_xy_grid_coord[:,0], fov_coverage_xy_grid_coord[:,1], \
                s=5, color='blue', marker='.', alpha=0.05, label='radar fov coverage')

        ax.scatter(0, 0, s=150, marker='*', color='black', label='ego vehicle wheel base centre')
        self.stationary_meas = ax.scatter([],[], s=0.1, color='red', marker='.', label='stationary meas')
        self.poses = ax.scatter([],[], s=4, color='blue', marker='.', label='ego vehicle trajectory')
        ax.legend(loc='upper right')
        self.ax = ax
        

    def __call__(self, t):

        # radar measurement accumulation
        self.meas_hist, \
        self.T_prev, \
        timestamp_rad\
             = meas_integration_step(
                    self.data_dir, 
                    self.data_odom, 
                    self.frame_offset + t,
                    self.trigger,
                    self.meas_hist,
                    self.T_prev)
        self.trigger = False

        # ego vehicle poses w.r.t the curremnt frame
        T = construct_SE2_group_element(
            self.meas_hist.pose_x[self.meas_hist.top], 
            self.meas_hist.pose_y[self.meas_hist.top], 
            self.meas_hist.pose_yaw[self.meas_hist.top])
        x_loc, y_loc = coordinate_transform_meas(self.meas_hist.pose_x, self.meas_hist.pose_y, inverse_SE2(T))

        # plot the cloud
        meas = np.stack([self.meas_hist.meas_xcoord, self.meas_hist.meas_ycoord], axis=-1)
        poses = np.stack([x_loc, y_loc], axis=-1)
        self.stationary_meas.set_offsets(meas)
        self.poses.set_offsets(poses)
        self.ax.set_title('Scene ' + str(self.scene) + ': Radar scan accumulation w.r.t the ego vehicle at time ' + f'{timestamp_rad:.2f}' + ' sec', fontsize=14)




fig, ax = plt.subplots()
ax = set_plot_properties(ax, x_min=-85, x_max=105, y_min=-100, y_max=100)
integrated_scans = animate(ax)
anim = FuncAnimation(fig, integrated_scans, frames=np.arange(n), interval=1, repeat=False)
plt.show()


