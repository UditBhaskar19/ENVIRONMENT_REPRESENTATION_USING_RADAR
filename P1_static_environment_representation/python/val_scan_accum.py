# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validation script lib_scan_accum.py
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from lib_datastruct import meas_hist_buffer
from lib_grid import convert_meas_polar_to_cartesian
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
    sync_radar_with_egomotion,
    construct_SE2_group_element,
    ego_compensate_prev_meas_vehicle_frame,
    coordinate_transform_meas, 
    inverse_SE2   
)


def plot_meas(meas_x, meas_y, meas_x_sync, meas_y_sync):
    _, ax = plt.subplots(1,1)
    ax.scatter(meas_x, meas_y, s=10, color='red', marker='o', label='radar measurements')
    ax.scatter(meas_x_sync, meas_y_sync, s=10, color='blue', marker='o', label='radar measurements synced with ego-motion')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.title('radar measurements / radar measurements synced', fontsize=16)


def plot_integrated_scans(meas_x, meas_y):
    _, ax = plt.subplots(1,1)
    ax.scatter(meas_x, meas_y, s=1, color='red', marker='.', label='radar measurements')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.title('radar measurements / radar measurements synced', fontsize=16)


def plot_integrated_scans_and_ego_pose(meas_x, meas_y, x_loc, y_loc, star_size):
    _, ax = plt.subplots(1,1)
    ax.scatter(meas_x, meas_y, s=1, color='red', marker='.', label='radar measurements')
    ax.plot(x_loc, y_loc, '-', linewidth=2, color='blue', label='ego trajectory')
    ax.scatter(0, 0, s=star_size, marker='*', color='black', label='ego vehicle current location')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.title('radar measurements / radar measurements synced', fontsize=16)

# --------------------------------------------------------------------------------------------------------------------

def validate_sync_func(scene, frameid):

    data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
    file_timestamp_and_odom = os.path.join(data_dir, 'timestamp_and_odom.csv')
    _, data_timestamp_and_odom = parse_csv_file(file_timestamp_and_odom)

    # extract time stamps and odom
    odom_vx, _, odom_yawrate, _, _, _, \
    radar_id, timestamp_rad, timestamp_odom = extract_sensor_sync_data(data_timestamp_and_odom, frameid)

    # extract the radar frame data
    rad_meas = extract_radar_data(data_dir, frameid)
    mount_param = lib_const.radars_mount[radar_id]
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
    meas_sync_x, \
    meas_sync_y  \
        = sync_radar_with_egomotion(
            meas_stationary_x, meas_stationary_y, timestamp_rad, 
            odom_vx, odom_yawrate, timestamp_odom)

    print(timestamp_rad - timestamp_odom)
    plot_meas(meas_stationary_x, meas_stationary_y, meas_sync_x, meas_sync_y)

# --------------------------------------------------------------------------------------------------------------------

def meas_integration_step(
    data_dir, 
    odom_data, 
    frameid,
    first_time,
    meas_sync_x,
    meas_sync_y,
    T_prev,
    X,Y,
    ):

    # extract time stamps and odom
    odom_vx, _, odom_yawrate, x_loc, y_loc, yaw_loc, \
    radar_id, timestamp_rad, timestamp_odom = extract_sensor_sync_data(odom_data, frameid)
    X.append(x_loc); Y.append(y_loc)

    # extract the radar frame data
    rad_meas = extract_radar_data(data_dir, frameid)
    mount_param = lib_const.radars_mount[radar_id]
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
    if first_time == True: 
        T_prev = construct_SE2_group_element(x_loc, y_loc, yaw_loc)
        meas_sync_x = np.concatenate([meas_sync_x, meas_stationary_x], axis=0)
        meas_sync_y = np.concatenate([meas_sync_y, meas_stationary_y], axis=0)

    else:
        T_curr = construct_SE2_group_element(x_loc, y_loc, yaw_loc)
        meas_sync_x, \
        meas_sync_y \
            = ego_compensate_prev_meas_vehicle_frame(
                meas_sync_x, meas_sync_y,
                T_curr, T_prev)
        T_prev = T_curr

        # for plotting
        meas_sync_x = np.concatenate([meas_sync_x, meas_stationary_x], axis=0)
        meas_sync_y = np.concatenate([meas_sync_y, meas_stationary_y], axis=0)

    return meas_sync_x, meas_sync_y, T_prev, X, Y

# --------------------------------------------------------------------------------------------------------------------

def meas_integration_step_v2(
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

    return meas_hist, T_prev

# --------------------------------------------------------------------------------------------------------------------

def integrate_radar_scans_all(scene, frame_offset, num_frames):

    data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
    file_timestamp_and_odom = os.path.join(data_dir, 'timestamp_and_odom.csv')
    _, data_timestamp_and_odom = parse_csv_file(file_timestamp_and_odom)
    num_samples = count_csv_files(data_dir) - 1

    # ego poses
    x_loc = []
    y_loc = []

    # for plotting
    T_prev = 0
    trigger = True
    meas_sync_x = np.array([])
    meas_sync_y = np.array([])

    frame_offset = 0
    num_frames = num_samples

    for i in range(num_frames):

        print(i)

        frameid = frame_offset + i

        meas_sync_x,    \
        meas_sync_y,    \
        T_prev,         \
        x_loc, y_loc    \
            = meas_integration_step(
                data_dir, data_timestamp_and_odom, 
                frameid, trigger,
                meas_sync_x, meas_sync_y,
                T_prev, x_loc, y_loc)
        trigger = False

    x_loc, y_loc = coordinate_transform_meas(x_loc, y_loc, inverse_SE2(T_prev))
    plot_integrated_scans_and_ego_pose(meas_sync_x, meas_sync_y, x_loc, y_loc, star_size=200)

# --------------------------------------------------------------------------------------------------------------------

def integrate_n_radar_scans(scene, frame_offset, num_frames, buffer_size):

    data_dir = os.path.join(config.root_dir, scene, 'radars_sync')
    file_timestamp_and_odom = os.path.join(data_dir, 'timestamp_and_odom.csv')
    _, data_timestamp_and_odom = parse_csv_file(file_timestamp_and_odom)
    num_samples = count_csv_files(data_dir) - 1

    # for plotting
    T_prev = 0
    trigger = True
    meas_hist = meas_hist_buffer(buffer_size)

    # frame_offset = 0
    # num_frames = num_samples

    for i in range(num_frames):

        print(i)

        frameid = frame_offset + i

        meas_hist, \
        T_prev = meas_integration_step_v2(
            data_dir, 
            data_timestamp_and_odom, 
            frameid,
            trigger,
            meas_hist,
            T_prev)
        trigger = False

    T = construct_SE2_group_element(
        meas_hist.pose_x[meas_hist.top], 
        meas_hist.pose_y[meas_hist.top], 
        meas_hist.pose_yaw[meas_hist.top])

    x_loc, y_loc = coordinate_transform_meas(meas_hist.pose_x, meas_hist.pose_y, inverse_SE2(T))
    plot_integrated_scans_and_ego_pose(meas_hist.meas_xcoord, meas_hist.meas_ycoord, x_loc, y_loc, star_size=200)

# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    print('Validating scan_integration.py !!!!!! ........')

    scene = '105'

    frame_offset = 1160
    num_frames = 1000
    buffer_size = 1000
    buffer_size = np.clip(buffer_size, 1, num_frames)

    integrate_n_radar_scans(scene, frame_offset, num_frames, buffer_size)
    # integrate_radar_scans_all(scene, frame_offset, num_frames)

    plt.show()

# --------------------------------------------------------------------------------------------------------------------


