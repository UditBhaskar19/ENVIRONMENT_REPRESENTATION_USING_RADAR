# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : File reading utilities
# =====================================================================================================================

import lib_const
import csv, os
import numpy as np

def parse_csv_file(file, header=True):
    data = []
    fields = []
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)      
        if header:
            fields = next(csvreader)             
        for row in csvreader:                
            data.append(row)
        data = np.array(data, dtype = np.float64)
    return fields, data


def count_csv_files(files_dir):
    num_files = 0
    for fname in os.listdir(files_dir):
        if fname.endswith('.csv'): 
            num_files += 1
    return num_files


def select_radar_mount_parameters(radar_id):
    if radar_id == 'radar1': mount = lib_const.radars_mount[0, :]
    elif radar_id == 'radar2': mount = lib_const.radars_mount[1, :]
    elif radar_id == 'radar3': mount = lib_const.radars_mount[2, :]
    elif radar_id == 'radar4': mount = lib_const.radars_mount[3, :]
    else : print('Invalid Option !! ... '); return 0, 0, 0
    tx = mount[0]
    ty = mount[1]
    azimuth = mount[2]
    return tx, ty, azimuth


def extract_sensor_sync_data(sensor_data, idx):
    odom_vx = sensor_data[idx, lib_const.sync_attr['vx_odom']]
    odom_vy = 0.0
    odom_yawrate = sensor_data[idx, lib_const.sync_attr['yawrate_odom']]
    radar_id = int(sensor_data[idx, lib_const.sync_attr['radar_id']]) - 1
    timestamp_rad = sensor_data[idx, lib_const.sync_attr['timestamp_rad']]
    timestamp_odom = sensor_data[idx, lib_const.sync_attr['timestamp_odom']]
    x_loc = sensor_data[idx, lib_const.sync_attr['x_loc']]
    y_loc = sensor_data[idx, lib_const.sync_attr['y_loc']]
    yaw_loc = sensor_data[idx, lib_const.sync_attr['yaw_loc']]
    return (
        odom_vx, odom_vy, odom_yawrate, x_loc, y_loc, yaw_loc,
        radar_id, timestamp_rad, timestamp_odom
    )


def extract_sensor_sync_data_single_radar(sensor_data, idx):
    odom_vx = sensor_data[idx, lib_const.odom_attr['vx_odom']]
    odom_vy = 0.0
    odom_yawrate = sensor_data[idx, lib_const.odom_attr['yawrate_odom']]
    timestamp_rad = sensor_data[idx, lib_const.odom_attr['timestamp_rad']]
    timestamp_odom = sensor_data[idx, lib_const.odom_attr['timestamp_odom']]
    x_loc = sensor_data[idx, lib_const.odom_attr['x_loc']]
    y_loc = sensor_data[idx, lib_const.odom_attr['y_loc']]
    yaw_loc = sensor_data[idx, lib_const.odom_attr['yaw_loc']]
    return (
        odom_vx, odom_vy, odom_yawrate, x_loc, y_loc, yaw_loc,
        timestamp_rad, timestamp_odom
    )


def extract_radar_data(data_dir, idx):
    file_meas = os.path.join(data_dir, str(idx + 1) + '.csv')
    _, rad_meas = parse_csv_file(file_meas)
    return rad_meas


