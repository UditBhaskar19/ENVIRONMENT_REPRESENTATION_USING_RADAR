# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validation script lib_meas_selection.py
# =====================================================================================================================

import config, lib_const, os
import numpy as np
import matplotlib.pyplot as plt
from lib_read_data import (
    parse_csv_file, 
    select_radar_mount_parameters
)
from lib_meas_selection import (
    generate_meas_sensor_frame,
    select_stationary_measurements
)


def plot_predicted_range_rate_profile(vx_snsr, vy_snsr, z):
    meas_azi = z[:, 1]
    meas_vr = z[:, 2]
    
    min_azi = np.min(meas_azi)
    max_azi = np.max(meas_azi)
    azivals = np.linspace(start=min_azi, stop=max_azi, num=100)
    pred_vr = -( vx_snsr * np.cos(azivals) + vy_snsr * np.sin(azivals) )
    
    fig, ax = plt.subplots(1,1)
    ax.plot(lib_const.rad2deg*azivals, pred_vr, color='green', label='predicted range rate line')
    ax.plot(lib_const.rad2deg*azivals, pred_vr + config.gamma_stationary, '-.', color='black', label='upper gate boundary')
    ax.plot(lib_const.rad2deg*azivals, pred_vr - config.gamma_stationary, '-.', color='blue', label='lower gate boundary')
    ax.scatter(lib_const.rad2deg*meas_azi, meas_vr, s=20, color='red', marker='x', label='radar measurements')
    ax.legend(loc='upper right')
    ax.set_xlabel('azimuth (deg)')
    ax.set_ylabel('range-rate (m/s)')
    plt.title('gated range-rates with odometry ego-motion', fontsize=16)
    

def plot_selected_meas(z, z_stationary, vx_snsr, vy_snsr):
    meas_azi = z[:, 1]
    meas_vr = z[:, 2]
    
    min_azi = np.min(meas_azi)
    max_azi = np.max(meas_azi)
    azivals = np.linspace(start=min_azi, stop=max_azi, num=100)
    pred_vr = -( vx_snsr * np.cos(azivals) + vy_snsr * np.sin(azivals) )

    fig, ax = plt.subplots(1,1)
    ax.plot(lib_const.rad2deg*azivals, pred_vr + config.gamma_stationary, '-.', color='black', label='upper gate boundary')
    ax.plot(lib_const.rad2deg*azivals, pred_vr - config.gamma_stationary, '-.', color='black', label='lower gate boundary')
    ax.scatter(lib_const.rad2deg*meas_azi, meas_vr, s=20, color='red', marker='x', label='radar dynamic measurements')
    ax.scatter(lib_const.rad2deg*z_stationary[:, 1], z_stationary[:, 2], s=20, color='blue', marker='x', label='radar static measurements')
    ax.legend(loc='upper right')
    ax.set_xlabel('azimuth (deg)')
    ax.set_ylabel('range-rate (m/s)')
    plt.title('selected measurements by RANSAC', fontsize=16)

    
if __name__ == '__main__':

    print('Validating meas_selection.py !!!!!! ........')

    scene = '108'       # 105, 108 
    sensor = 'radar2'
    frameid = 10

    data_dir = os.path.join(config.root_dir, scene, sensor)
    file_timestamp_and_odom = os.path.join(data_dir, 'timestamp_and_odom.csv')
    _, data_timestamp_and_odom = parse_csv_file(file_timestamp_and_odom)                 # extract odometry data and timestamps
    radar_mount_param = select_radar_mount_parameters(sensor)                            # extract radar mount parameters
    

    # extract the radar frame data
    file_meas = os.path.join(data_dir, str(frameid + 1) + '.csv')
    _, rad_meas = parse_csv_file(file_meas)

    # extract time stamps and odom
    timestamp_rad = data_timestamp_and_odom[frameid, lib_const.odom_attr['timestamp_rad']]
    vx_odom = data_timestamp_and_odom[frameid, lib_const.odom_attr['vx_odom']]
    yawrate_odom = data_timestamp_and_odom[frameid, lib_const.odom_attr['yawrate_odom']]

    vy_ego = 0.0
    tx, ty, theta = radar_mount_param
    vx_sensor, vy_sensor = generate_meas_sensor_frame(vx_odom, vy_ego, yawrate_odom, tx, ty, theta)
    plot_predicted_range_rate_profile(vx_sensor, vy_sensor, rad_meas)

    meas_stationary = select_stationary_measurements(
        rad_meas, radar_mount_param,
        vx_odom, yawrate_odom)
    plot_selected_meas(rad_meas, meas_stationary, vx_sensor, vy_sensor)

    plt.show()



