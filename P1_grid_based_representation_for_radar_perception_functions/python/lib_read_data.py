# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : File reading and data extraction utilities
# =====================================================================================================================

import lib_const, config
import csv, os
import numpy as np

# =====================================================================================================================

def parse_csv_file(file, header=True):
    data = []
    fields = []
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)      
        if header:
            fields = next(csvreader)             
        for row in csvreader:                
            data.append(row)
        data = np.array(data, dtype = np.float32)
    return fields, data

# =====================================================================================================================

def scene_meta_info(root_dir, scene_name):

    scene_dir = os.path.join(root_dir, scene_name)

    front_rad_data_dir = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[0])
    left_rad_data_dir = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[1])
    rear_left_rad_data_dir = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[2])
    rear_right_rad_data_dir = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[3])
    right_rad_data_dir = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[4])

    front_rad_time_stamps = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[0], 'time_stamps_sec.csv')
    left_rad_time_stamps = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[1], 'time_stamps_sec.csv')
    rear_left_rad_time_stamps = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[2], 'time_stamps_sec.csv')
    rear_right_rad_time_stamps = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[3], 'time_stamps_sec.csv')
    right_rad_time_stamps = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[4], 'time_stamps_sec.csv')

    front_rad_calib_file = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[0] + '_calib_extrinsic.csv')
    left_rad_calib_file = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[1] + '_calib_extrinsic.csv')
    rear_left_rad_calib_file = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[2] + '_calib_extrinsic.csv')
    rear_right_rad_calib_file = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[3] + '_calib_extrinsic.csv')
    right_rad_calib_file = os.path.join(scene_dir, 'radar', lib_const.radar_location_attr[4] + '_calib_extrinsic.csv')

    rad_data_meta_info = {
        'front_rad_data_dir': front_rad_data_dir,
        'left_rad_data_dir': left_rad_data_dir,
        'rear_left_rad_data_dir': rear_left_rad_data_dir,
        'rear_right_rad_data_dir': rear_right_rad_data_dir,
        'right_rad_data_dir': right_rad_data_dir
    }

    timestamps_meta_info = {
        'front_rad_time_stamps': front_rad_time_stamps,
        'left_rad_time_stamps': left_rad_time_stamps,
        'rear_left_rad_time_stamps': rear_left_rad_time_stamps,
        'rear_right_rad_time_stamps': rear_right_rad_time_stamps,
        'right_rad_time_stamps': right_rad_time_stamps
    }

    calib_meta_info = {
        'front_rad_calib_file': front_rad_calib_file,
        'left_rad_calib_file': left_rad_calib_file,
        'rear_left_rad_calib_file': rear_left_rad_calib_file,
        'rear_right_rad_calib_file': rear_right_rad_calib_file,
        'right_rad_calib_file': right_rad_calib_file
    }

    return rad_data_meta_info, timestamps_meta_info, calib_meta_info

# =====================================================================================================================

def extract_single_radar_timestamp(sensor, timestamps_meta_info):
    _, rad_timestamp = parse_csv_file(timestamps_meta_info[sensor + '_time_stamps'])
    rad_timestamp = rad_timestamp.reshape(-1)
    return rad_timestamp

# =====================================================================================================================

def extract_single_radar_mount_info(sensor, calib_meta_info):
    calib_attr, rad_calib = parse_csv_file(calib_meta_info[sensor + '_calib_file'])
    rad_calib = rad_calib.reshape(-1)
    calib_attr = {calib_attr[i]:i for i in range(len(calib_attr))}
    Q = np.array([rad_calib[calib_attr['q1']], \
                  rad_calib[calib_attr['q2']], \
                  rad_calib[calib_attr['q3']], \
                  rad_calib[calib_attr['q4']]])
    _, _, rad_mount_yaw = convert_quaternion2eular_angles(Q)
    rad_mount_x = rad_calib[calib_attr['Tx']]
    rad_mount_y = rad_calib[calib_attr['Ty']]
    return (
        rad_mount_x, 
        rad_mount_y, 
        rad_mount_yaw
    )

# =====================================================================================================================

def extract_radar_mount_info(scene_name):
    mount_parameters = np.zeros((lib_const.num_radars, 3), dtype=np.float32)
    _, _, calib_meta_info = scene_meta_info(config.root_dir, scene_name)
    for i in range(lib_const.num_radars):
        radar = lib_const.radar_location_attr[i] + '_rad'
        mount_x, mount_y, mount_yaw = extract_single_radar_mount_info(radar, calib_meta_info)
        mount_parameters[i] = np.array([mount_x, mount_y, mount_yaw], dtype=np.float32)
    return mount_parameters

# =====================================================================================================================

def extract_radar_frame(
    sensor, idx, 
    rad_data_meta_info, 
    measurement_selection_function):
    """ load radar measurement from csv file """
    file = os.path.join(rad_data_meta_info[sensor + '_rad_data_dir'], str(idx) + '.csv')
    rad_meas_attr, rad_data = parse_csv_file(file)
    if rad_data.shape[0] > 0:
        rad_meas_attr = {rad_meas_attr[i]:i for i in range(len(rad_meas_attr))}
        rad_sel_meas = measurement_selection_function(rad_data, rad_meas_attr)
        z = rad_sel_meas[:, [rad_meas_attr['x'], rad_meas_attr['y'], rad_meas_attr['vx'], rad_meas_attr['vy']]]
        z_rms = rad_sel_meas[:, [rad_meas_attr['x_rms'], rad_meas_attr['y_rms'], rad_meas_attr['vx_rms'], rad_meas_attr['vy_rms']]]
        phd0 = rad_sel_meas[:, rad_meas_attr['pdh0']]
    else: 
        z = np.zeros((0, 4), dtype=np.float32)
        z_rms = np.zeros((0, 4), dtype=np.float32)
        phd0 = np.zeros((0, ), dtype=np.float32)
    return z, z_rms, phd0

# =====================================================================================================================

def extract_radar_data(
    frameid,
    rad_data_meta_info, 
    meas_selection_function, 
    frame_summary):
    
    # get the radar name
    radar_id = frame_summary[frameid, 1].astype(np.int16)
    radar_name = lib_const.radar_id_to_name[radar_id]

    # get the radar data
    frame_id = frame_summary[frameid, 2].astype(np.int16)
    rad_meas, _, phd0 \
        = extract_radar_frame(
            radar_name, frame_id, 
            rad_data_meta_info, 
            meas_selection_function)

    return ( rad_meas,  phd0)

# =====================================================================================================================

def create_rad_frame_summary(scene_name):
    data_dir = os.path.join(config.root_dir, scene_name)
    _, timestamps_meta_info, _ = scene_meta_info(config.root_dir, scene_name)

    # create radar frame info summary
    rad_timestamps = []
    rad_ids = []
    frame_ids = []
    for sensor in lib_const.radar_location_attr:
        timestamps = extract_single_radar_timestamp(sensor + '_rad', timestamps_meta_info)
        rad_timestamps.append(timestamps)
        rad_ids.append(np.repeat(lib_const.radar_id[sensor], timestamps.shape[0]))
        frame_ids.append(np.arange(1, timestamps.shape[0] + 1))

    rad_timestamps = np.concatenate(rad_timestamps, axis=0)
    rad_ids = np.concatenate(rad_ids, axis=0)
    frame_ids = np.concatenate(frame_ids, axis=0)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # sort the radar frames by time
    frame_summary = frame_summary[np.argsort(frame_summary[:, 0])]
    return frame_summary

# =====================================================================================================================

def convert_quaternion2eular_angles(Q):

    temp1 = 2 * (Q[0] * Q[1] + Q[2] * Q[3])
    temp2 = 1 - 2 * (Q[1] ** 2 + Q[2] ** 2)
    temp3 = 2 * (Q[0] * Q[2] - Q[3] * Q[1])
    temp4 = 2 * (Q[0] * Q[3] + Q[1] * Q[2])
    temp5 = 1 - 2 * (Q[2] ** 2 + Q[3] ** 2)

    roll = np.arctan2(temp1, temp2)
    pitch = np.arcsin(temp3)
    yaw = np.arctan2(temp4, temp5)

    return roll, pitch, yaw

# =====================================================================================================================

def select_stationary_high_confidence_meas_radar(
    radar_meas, 
    radar_attr):
    """ select stationary measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    condition1 = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==1 , 
        np.int16(dyn_prop_vals)==3 , 
        np.int16(dyn_prop_vals)==5
    )
    condition2 = np.int16(valid_meas_vals)==0
    condition = np.logical_and(condition1, condition2)
    return radar_meas[condition, :]

# =====================================================================================================================

def select_moving_high_confidence_meas_radar(
    radar_meas, 
    radar_attr):
    """ select stationary measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    condition1 = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==0 , 
        np.int16(dyn_prop_vals)==2 , 
        np.int16(dyn_prop_vals)==6
    )
    condition2 = np.int16(valid_meas_vals)==0
    condition = np.logical_and(condition1, condition2)
    return radar_meas[condition, :]

# =====================================================================================================================

def select_stationary_all_valid_meas_radar(
    radar_meas, 
    radar_attr):
    """ select stationary measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    condition1 = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==1 , 
        np.int16(dyn_prop_vals)==3 , 
        np.int16(dyn_prop_vals)==5
    )

    condition2 = np.zeros((valid_meas_vals.shape[0], ), dtype=np.bool8)
    is_equal_vals = [0,4,8,9,10,11,12,15,16,17]
    for val in is_equal_vals:
        condition2 = np.logical_or(condition2, np.int16(valid_meas_vals)==val)

    condition = np.logical_and(condition1, condition2)
    return radar_meas[condition, :]

# =====================================================================================================================

def select_moving_all_valid_meas_radar(
    radar_meas, 
    radar_attr):
    """ select stationary measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    condition1 = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==0 , 
        np.int16(dyn_prop_vals)==2 , 
        np.int16(dyn_prop_vals)==6
    )

    condition2 = np.zeros((valid_meas_vals.shape[0], ), dtype=np.bool8)
    is_equal_vals = [0,4,8,9,10,11,12,15,16,17]
    for val in is_equal_vals:
        condition2 = np.logical_or(condition2, np.int16(valid_meas_vals)==val)
    
    condition = np.logical_and(condition1, condition2)
    return radar_meas[condition, :]

# =====================================================================================================================

def select_moving_all_meas_radar(
    radar_meas, 
    radar_attr):
    """ select stationary measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    condition = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==0 , 
        np.int16(dyn_prop_vals)==2 , 
        np.int16(dyn_prop_vals)==6
    )
    return radar_meas[condition, :]

# =====================================================================================================================

def select_all_invalid_meas_radar(
    radar_meas, 
    radar_attr):

    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    ambig_state = radar_meas[:, radar_attr['ambig_state']]
    pdh0 = radar_meas[:, radar_attr['pdh0']]
    condition = np.zeros((valid_meas_vals.shape[0], ), dtype=np.bool8)
    
    is_equal_vals = [1,2,3,6,7,14]
    for val in is_equal_vals:
        condition = np.logical_or(condition, np.int16(valid_meas_vals)==val)

    is_equal_vals = [0,1]
    for val in is_equal_vals:
        condition = np.logical_or(condition, np.int16(ambig_state)==val)

    condition = np.logical_and(condition, pdh0 != 0 )   
    return radar_meas[condition, :]

# =====================================================================================================================

if __name__ == '__main__':

    print(extract_radar_mount_info('scene-0061'))
    print(extract_radar_mount_info('scene-0103'))
    print(extract_radar_mount_info('scene-0655'))
    print(extract_radar_mount_info('scene-0796'))
    print(extract_radar_mount_info('scene-0916'))
    print(extract_radar_mount_info('scene-1077'))
    print(extract_radar_mount_info('scene-1094'))



    rad_data_meta_info, timestamps_meta_info, calib_meta_info = scene_meta_info(config.root_dir, 'scene-0061')
    print(rad_data_meta_info)
    print(timestamps_meta_info)
    print(calib_meta_info)