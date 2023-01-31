# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Constants
# =====================================================================================================================

import numpy as np

# radar fov boundary points
radar_fov_ranges = np.array([100, 100])
radar_fov_azimuths = np.array([-70, 70])

# mount info in x, y, angle
radar1_mount = np.array([3.663, -0.873, -1.48418552])
radar2_mount = np.array([3.86, -0.7, -0.436185662])
radar3_mount = np.array([3.86, 0.7, 0.436])
radar4_mount = np.array([3.663, 0.873, 1.484])
radars_mount = np.stack([radar1_mount, radar2_mount, radar3_mount, radar4_mount], axis=0)
num_radars = radars_mount.shape[0]

# constants
rad2deg = 180/np.pi
deg2rad = np.pi/180
eps = 1e-10

# mapping from column index to field attributes in csv files
# radar measurement
rad_meas_attr = {}
rad_meas_attr['range'] = 0
rad_meas_attr['azimuth'] = 1
rad_meas_attr['range_rate'] = 2
rad_meas_attr['rcs'] = 3

# timestamp and odom attr
odom_attr = {}
odom_attr['timestamp_rad'] = 0
odom_attr['timestamp_odom'] = 1
odom_attr['x_loc'] = 2
odom_attr['y_loc'] = 3
odom_attr['yaw_loc'] = 4
odom_attr['vx_odom'] = 5
odom_attr['yawrate_odom'] = 6

# radars and odom sync attr
sync_attr = {}
sync_attr['timestamp_rad'] = 0
sync_attr['timestamp_odom'] = 1
sync_attr['radar_id'] = 2
sync_attr['x_loc'] = 3
sync_attr['y_loc'] = 4
sync_attr['yaw_loc'] = 5
sync_attr['vx_odom'] = 6
sync_attr['yawrate_odom'] = 7


