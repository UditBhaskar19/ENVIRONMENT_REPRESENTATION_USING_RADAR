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
from lib_meas_selection import ( 
    coordinate_transform_px_py, 
    coordinate_transform_vx_vy
)
from lib_read_data import (
    select_radar_mount_parameters,
    parse_csv_file, count_csv_files, 
    extract_sensor_sync_data_single_radar, 
    extract_radar_data
)


scene = '105'
sensor = 'radar3'
data_dir = os.path.join(config.root_dir, scene, sensor)
_, data_odom = parse_csv_file(os.path.join(data_dir, 'timestamp_and_odom.csv'))
num_samples = count_csv_files(data_dir) - 1
frame_offset = 0
n = num_samples

GridCartesian = grid_properties(
        min_x = 0.0, 
        max_x = 120.0, 
        min_y = -60.0, 
        max_y = 60.0, 
        dx = 0.2, 
        dy = 0.2
    )

fov_coverage = FOV_grid_coverage(GridCartesian)
xy_grid_coord = np.stack([fov_coverage.x_coord, fov_coverage.y_coord], axis=-1)

# ===================================================================================================================

def get_radar_id(sensor):
    if sensor == 'radar1': return 0
    elif sensor == 'radar2': return 1
    elif sensor == 'radar3': return 2
    elif sensor == 'radar4': return 3
    else : return -1

# ===================================================================================================================

def meas_extraction_step(
    data_dir, 
    odom_data,
    frameid,
    radarid):

    # extract time stamps and odom and radar mount parameters
    _, _, _, _, _, _, timestamp_rad, _ = extract_sensor_sync_data_single_radar(odom_data, frameid)
    tx, ty, theta = select_radar_mount_parameters(radarid)

    # extract the radar frame and the doppler
    rad_meas = extract_radar_data(data_dir, frameid)
    rad_meas_azi = rad_meas[:, lib_const.rad_meas_attr['azimuth']]
    rad_meas_vr = rad_meas[:, lib_const.rad_meas_attr['range_rate']]
    rad_meas_vrx = rad_meas_vr * np.cos( rad_meas_azi )
    rad_meas_vry = rad_meas_vr * np.sin( rad_meas_azi )
    
    # polar to cartesian conversion for all meas
    rad_meas_x, rad_meas_y \
        = convert_meas_polar_to_cartesian(
            rad_meas[:, lib_const.rad_meas_attr['range']], 
            rad_meas[:, lib_const.rad_meas_attr['azimuth']])

    # coordinate transformation for visualization
    rad_meas_x, rad_meas_y = coordinate_transform_px_py(rad_meas_x, rad_meas_y, tx, ty, theta)
    rad_meas_vrx, rad_meas_vry = coordinate_transform_vx_vy(rad_meas_vrx, rad_meas_vry, theta)

    print('frameid: ', frameid,\
         '     num_meas: ',  rad_meas.shape[0],\
         '     time: ', f'{timestamp_rad:.3f}')

    return (
        rad_meas_x, rad_meas_y,
        rad_meas_vrx, rad_meas_vry,
        timestamp_rad, radarid)

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
        self.sensor = sensor
        self.data_dir = data_dir
        self.data_odom = data_odom
        self.num_samples = num_samples
        self.frame_offset = frame_offset
        self.radar_id = get_radar_id(sensor) 

        # radar fov coordinates
        self.xy_grid_coord = xy_grid_coord
        self.fov_coverage_flag = fov_coverage.fov_coverage_flag
        self.radar_fov = ax.scatter([],[], s=5, color='blue', marker='.', alpha=0.1, label='active radar fov')

        self.meas_xy = ax.scatter([],[], s=45, color='red', marker='x', label='radar measurements')
        self.meas_vr = ax.quiver([], [], [], [])
        ax.legend(loc='upper right')
        self.ax = ax

    def __call__(self, t):

        rad_meas_x, rad_meas_y,\
        rad_meas_vrx, rad_meas_vry, \
        timestamp_rad, radarid \
            = meas_extraction_step(
                self.data_dir, 
                self.data_odom, 
                self.frame_offset + t,
                self.sensor)

        self.radar_fov.set_offsets(self.xy_grid_coord[self.fov_coverage_flag[self.radar_id]])
        radar_meas_all = np.stack([rad_meas_x, rad_meas_y], axis=-1)
        self.meas_xy.set_offsets(radar_meas_all)
        self.meas_vr.remove()
        self.meas_vr = ax.quiver( rad_meas_x, rad_meas_y, rad_meas_vrx, rad_meas_vry, color='k', width=0.002)
        self.ax.set_title('Scene ' + str(self.scene) + ': ' + radarid + \
                          ' scan visualization w.r.t the ego vehicle at time ' + f'{timestamp_rad:.2f}' + ' sec', \
                          fontsize=14)        

# ===================================================================================================================

fig, ax = plt.subplots()
ax = set_plot_properties(ax, x_min=0, x_max=105, y_min=-20, y_max=40)
integrated_scans = animate(ax)
anim = FuncAnimation(fig, integrated_scans, frames=np.arange(n), interval=1, repeat=False)
plt.show()

