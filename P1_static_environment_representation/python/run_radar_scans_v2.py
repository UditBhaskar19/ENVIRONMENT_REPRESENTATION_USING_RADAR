# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : ANIMATION script for visualizing radar scans
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib_grid import (
    convert_meas_polar_to_cartesian, 
    grid_properties, FOV_grid_coverage
)
from lib_read_data import (
    parse_csv_file, count_csv_files, 
    extract_sensor_sync_data, extract_radar_data
)
from lib_meas_selection import (
    identify_stationary_measurements, 
    coordinate_transform_px_py
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

def meas_extraction_step(
    data_dir, 
    odom_data, 
    frameid):

    # extract time stamps and odom
    odom_vx, _, odom_yawrate, _, _, _, \
    radar_id, timestamp_rad, _ = extract_sensor_sync_data(odom_data, frameid)

    # extract the radar frame and radar mount data
    rad_meas = extract_radar_data(data_dir, frameid)
    mount_param = lib_const.radars_mount[radar_id]

    # identify stationary measurements
    flag_stationary = identify_stationary_measurements(
        rad_meas, (mount_param[0], mount_param[1], mount_param[2]), 
        odom_vx, odom_yawrate)

    # polar to cartesian conversion for all meas
    rad_meas_x, \
    rad_meas_y \
        = convert_meas_polar_to_cartesian(
            rad_meas[:, lib_const.rad_meas_attr['range']], 
            rad_meas[:, lib_const.rad_meas_attr['azimuth']])

    # coordinate transformation from sensor frame to vehicle frame for all meas
    rad_meas_x, \
    rad_meas_y \
        = coordinate_transform_px_py(
            rad_meas_x, rad_meas_y, 
            mount_param[0], mount_param[1], mount_param[2])

    # seperate out the static and dynamic meas
    radar_meas_all = np.stack([rad_meas_x, rad_meas_y], axis=-1)
    radar_meas_static = radar_meas_all[flag_stationary]
    radar_meas_non_static = radar_meas_all[np.logical_not(flag_stationary)]

    print('frameid: ', frameid,\
         '     radar id: ',  radar_id + 1,\
         '     num_meas: ',  radar_meas_all.shape[0],\
         '     time: ', f'{timestamp_rad:.3f}')

    return (
        radar_meas_static, 
        radar_meas_non_static,
        timestamp_rad,
        radar_id )

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

        self.non_stationary_meas = ax.scatter([],[], s=20, color='green', marker='x', label='non stationary meas')
        self.stationary_meas = ax.scatter([],[], s=5, color='red', marker='x', label='stationary meas')
        ax.legend(loc='upper right')
        self.ax = ax

    def __call__(self, t):

        radar_meas_static, \
        radar_meas_non_static, \
        timestamp_rad, \
        radar_id \
            = meas_extraction_step(
                self.data_dir, 
                self.data_odom, 
                self.frame_offset + t)

        self.non_stationary_meas.set_offsets(radar_meas_non_static)
        self.stationary_meas.set_offsets(radar_meas_static)
        self.ax.set_title('Scene ' + str(self.scene) + ': Radar ' + str(radar_id) + \
                          ' scan visualization w.r.t the ego vehicle at time ' + f'{timestamp_rad:.2f}' + ' sec', \
                          fontsize=14)

# ===================================================================================================================

fig, ax = plt.subplots()
ax = set_plot_properties(ax, x_min=-85, x_max=105, y_min=-100, y_max=100)
integrated_scans = animate(ax)
anim = FuncAnimation(fig, integrated_scans, frames=np.arange(n), interval=0.1, repeat=False)
plt.show()


