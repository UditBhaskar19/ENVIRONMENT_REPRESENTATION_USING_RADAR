# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : sync radar frames with imu : associate each of the radar frames with the temporally closest imu measurement

# function names :
# - sync_imu_single_radar() : synchronize the imu with a single radar
# - sync_imu_all_radars() : synchronize the imu with all the radars
# - create_imu_summary() : create the required imu summary
# - create_pose_summary() : create the required pose summary
# - sync_ego_sensor_all_attr_single_radar() : synchronize the imu or ego pose with a single radar
# - sync_ego_sensor_all_attr_all_radars() : synchronize the imu or ego pose with all the radars
# - sync_cam_all_radars() : synchronize cameras with all radars by associating each of the radar frames with the temporally closest camera frame
# =====================================================================================================================

import numpy as np
import config, os, const
from lib_functions import convert_quaternion2eular_angles
from lib_read_data import (
    load_can_signal, 
    scene_rad_meta_info, 
    scene_cam_meta_info,
    extract_timestamp_single_sensor,
    extract_timestamp_all_sensors
)

# =====================================================================================================================

def sync_imu_single_radar(scene_name, radar_name):
    """ synchronize the imu with a single radar by associating each of the radar frames with the temporally closest imu measurement
    inputs : scene_name, radar_name - strings 
    outputs : frame_summary - A numpy array of shape ( n, 4 ), where n is the number of frames of the specific sensor 'radar_name'
                              Each entry in the array consists of (timestamp, rad_id, frame_id, closest_imu_meas_idx).
                              These are sorted entries in the increasing order of the timestamps
            : imu_summary - A numpy array of shape ( m, 2 ), where m is the length of the sequence of the imu measurement 
                            corrosponding to the scene 'scene_name'. Each entry in the array consists of (timestamp, ego_yaw).
                            these are sorted entries in the increasing order of the timestamps
    """
    data_dir = os.path.join(config.root_dir_rad, scene_name)
    can_data, can_attr = load_can_signal(data_dir)
    _, timestamps_meta_info, _ = scene_rad_meta_info(config.root_dir_rad, scene_name)

    # rad_timestamps = extract_single_radar_timestamp(radar_name + '_rad', timestamps_meta_info)
    rad_timestamps = extract_timestamp_single_sensor(radar_name , timestamps_meta_info)
    rad_ids = np.repeat(const.radar_id[radar_name], rad_timestamps.shape[0])
    frame_ids = np.arange(1, rad_timestamps.shape[0] + 1)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # create imu summary
    imu_timestamp = can_data['imu_data'][:, can_attr['imu_fields']['timestamp_sec']]
    imu_yaw_rate = can_data['imu_data'][:, can_attr['imu_fields']['rot_rate_z']]
    imu_summary = np.stack([imu_timestamp, imu_yaw_rate], axis=-1)

    # find the imu index temporally closest to a radar frame idx
    timestamps_radars = np.expand_dims(frame_summary[:, 0], axis=-1)
    timestamps_imu = np.expand_dims(imu_summary[:, 0], axis=0)
    dt = np.abs(timestamps_radars - timestamps_imu)
    closest_imu_meas = np.argmin(dt, axis=1)

    # frame_summary
    closest_imu_meas = np.expand_dims(closest_imu_meas, axis=-1)
    frame_summary = np.concatenate([frame_summary, closest_imu_meas], axis=-1)
    return frame_summary, imu_summary

# =====================================================================================================================

def sync_imu_all_radars(scene_name):
    """ synchronize the imu with all the radars by associating each of the radar frames with the temporally closest imu measurement
    inputs : scene_name - strings 
    outputs : frame_summary - A numpy array of shape ( n, 4 ), where n is total number of frames of all the radars in the scene 'scene_name'
                              Each entry in the array consists of (timestamp, rad_id, frame_id, closest_imu_meas_idx).
                              These are sorted entries in the increasing order of the timestamps
            : imu_summary - A numpy array of shape ( m, 2 ), where m is the length of the sequence of the imu measurement 
                            corrosponding to the scene 'scene_name'. Each entry in the array consists of (timestamp, ego_yaw).
                            these are sorted entries in the increasing order of the timestamps
    """
    data_dir = os.path.join(config.root_dir_rad, scene_name)
    can_data, can_attr = load_can_signal(data_dir)
    _, timestamps_meta_info, _ = scene_rad_meta_info(config.root_dir_rad, scene_name)

    # create radar frame info summary
    rad_timestamps = []
    rad_ids = []
    frame_ids = []
    for sensor in const.radar_location_attr:
        timestamps = extract_timestamp_single_sensor(sensor , timestamps_meta_info)
        rad_timestamps.append(timestamps)
        rad_ids.append(np.repeat(const.radar_id[sensor], timestamps.shape[0]))
        frame_ids.append(np.arange(1, timestamps.shape[0] + 1))

    rad_timestamps = np.concatenate(rad_timestamps, axis=0)
    rad_ids = np.concatenate(rad_ids, axis=0)
    frame_ids = np.concatenate(frame_ids, axis=0)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # create imu summary
    imu_timestamp = can_data['imu_data'][:, can_attr['imu_fields']['timestamp_sec']]
    imu_yaw_rate = can_data['imu_data'][:, can_attr['imu_fields']['rot_rate_z']]
    imu_summary = np.stack([imu_timestamp, imu_yaw_rate], axis=-1)

    # sort the radar frames by time
    frame_summary = frame_summary[np.argsort(frame_summary[:, 0])]

    # find the imu index temporally closest to a radar frame idx
    timestamps_radars = np.expand_dims(frame_summary[:, 0], axis=-1)
    timestamps_imu = np.expand_dims(imu_summary[:, 0], axis=0)
    dt = np.abs(timestamps_radars - timestamps_imu)
    closest_imu_meas = np.argmin(dt, axis=1)

    # frame_summary
    closest_imu_meas = np.expand_dims(closest_imu_meas, axis=-1)
    frame_summary = np.concatenate([frame_summary, closest_imu_meas], axis=-1)
    return frame_summary, imu_summary

# =====================================================================================================================

def create_imu_summary(can_data, can_attr):
    """ create the required imu summary """
    # timestamp_sec,ax,ay,az,rot_rate_x,rot_rate_y,rot_rate_z,q1,q2,q3,q4
    imu_timestamp = can_data['imu_data'][:, can_attr['imu_fields']['timestamp_sec']]
    imu_ax = can_data['imu_data'][:, can_attr['imu_fields']['ax']]
    imu_ay = can_data['imu_data'][:, can_attr['imu_fields']['ay']]
    imu_az = can_data['imu_data'][:, can_attr['imu_fields']['az']]
    imu_rot_rate_x = can_data['imu_data'][:, can_attr['imu_fields']['rot_rate_x']]
    imu_rot_rate_y = can_data['imu_data'][:, can_attr['imu_fields']['rot_rate_y']]
    imu_rot_rate_z = can_data['imu_data'][:, can_attr['imu_fields']['rot_rate_z']]
    # imu_q1 = can_data['imu_data'][:, can_attr['imu_fields']['q1']]
    # imu_q2 = can_data['imu_data'][:, can_attr['imu_fields']['q2']]
    # imu_q3 = can_data['imu_data'][:, can_attr['imu_fields']['q3']]
    # imu_q4 = can_data['imu_data'][:, can_attr['imu_fields']['q4']]
    
    imu_summary = np.stack([
        imu_timestamp, 
        imu_ax, imu_ay, imu_az, 
        imu_rot_rate_x, imu_rot_rate_y, imu_rot_rate_z], axis=-1)

    return imu_summary

# =====================================================================================================================

def create_pose_summary(can_data, can_attr):
    """ create the required pose summary """
    # timestamp_sec,px,py,pz,q1,q2,q3,q4,vx,vy,vz,ax,ay,az,rot_rate_x,rot_rate_y,rot_rate_z
    pose_timestamp = can_data['pose_data'][:, can_attr['pose_fields']['timestamp_sec']]
    pose_px = can_data['pose_data'][:, can_attr['pose_fields']['px']]
    pose_py = can_data['pose_data'][:, can_attr['pose_fields']['py']]
    pose_pz = can_data['pose_data'][:, can_attr['pose_fields']['pz']]

    # pose_vx = can_data['pose_data'][:, can_attr['pose_fields']['vx']]
    # pose_vy = can_data['pose_data'][:, can_attr['pose_fields']['vy']]
    # pose_vz = can_data['pose_data'][:, can_attr['pose_fields']['vz']]

    # pose_ax = can_data['pose_data'][:, can_attr['pose_fields']['ax']]
    # pose_ay = can_data['pose_data'][:, can_attr['pose_fields']['ay']]
    # pose_az = can_data['pose_data'][:, can_attr['pose_fields']['az']]

    # pose_rot_rate_x = can_data['pose_data'][:, can_attr['pose_fields']['rot_rate_x']]
    # pose_rot_rate_y = can_data['pose_data'][:, can_attr['pose_fields']['rot_rate_y']]
    # pose_rot_rate_z = can_data['pose_data'][:, can_attr['pose_fields']['rot_rate_z']]

    pose_q1 = can_data['pose_data'][:, can_attr['pose_fields']['q1']]
    pose_q2 = can_data['pose_data'][:, can_attr['pose_fields']['q2']]
    pose_q3 = can_data['pose_data'][:, can_attr['pose_fields']['q3']]
    pose_q4 = can_data['pose_data'][:, can_attr['pose_fields']['q4']]

    # convert the quarternion to eular
    Q = np.stack([pose_q1, pose_q2, pose_q3, pose_q4], axis=0)
    pose_roll, pose_pitch, pose_yaw = convert_quaternion2eular_angles(Q)
    
    pose_summary = np.stack([
        pose_timestamp, 
        pose_px, pose_py, pose_pz, 
        pose_roll, pose_pitch, pose_yaw], axis=-1)

    return pose_summary

# =====================================================================================================================

def sync_ego_sensor_all_attr_single_radar(scene_name, radar_name, sync_type):
    """ synchronize the imu or ego pose with a single radar by associating each of the radar frames with the temporally closest imu measurement
    inputs : scene_name, radar_name - strings 
           : sync_type - string to indicate if the ego sensor type is 'imu' or 'pose'
    outputs : frame_summary - A numpy array of shape ( n, 4 ), where n is total number of frames of all the radars in the scene 'scene_name'
                              Each entry in the array consists of (timestamp, rad_id, frame_id, closest_imu_meas_idx).
                              These are sorted entries in the increasing order of the timestamps
            : ego_summary - A numpy array of shape ( m, k ), where m is the length of the sequence of the ego measurement 
                            corrosponding to the scene 'scene_name'. Each entry in the array consists of (timestamp, ...).
                            these are sorted entries in the increasing order of the timestamps
    """
    data_dir = os.path.join(config.root_dir_rad, scene_name)
    can_data, can_attr = load_can_signal(data_dir)
    _, timestamps_meta_info, _ = scene_rad_meta_info(config.root_dir_rad, scene_name)

    #rad_timestamps = extract_single_radar_timestamp(radar_name + '_rad', timestamps_meta_info)
    rad_timestamps = extract_timestamp_single_sensor(radar_name , timestamps_meta_info)
    rad_ids = np.repeat(const.radar_id[radar_name], rad_timestamps.shape[0])
    frame_ids = np.arange(1, rad_timestamps.shape[0] + 1)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # create imu pr pose summary
    if sync_type == 'imu': ego_summary = create_imu_summary(can_data, can_attr)
    else: ego_summary = create_pose_summary(can_data, can_attr)

    # find the imu index temporally closest to a radar frame idx
    timestamps_radars =  np.expand_dims(frame_summary[:, 0], axis=-1)
    timestamps_ego = np.expand_dims(ego_summary[:, 0], axis=0)
    dt = np.abs(timestamps_radars - timestamps_ego)
    closest_ego_meas = np.argmin(dt, axis=1)

    # frame_summary
    closest_ego_meas = np.expand_dims(closest_ego_meas, axis=-1)
    frame_summary = np.concatenate([frame_summary, closest_ego_meas], axis=-1)
    return frame_summary, ego_summary

# =====================================================================================================================

def sync_ego_sensor_all_attr_all_radars(scene_name, sync_type):
    """ synchronize the imu or ego pose with all the radars by associating each of the radar frames with the temporally closest imu measurement
    inputs : scene_name, radar_name - strings 
           : sync_type - string to indicate if the ego sensor type is 'imu' or 'pose'
    outputs : frame_summary - A numpy array of shape ( n, 4 ), where n is total number of frames of all the radars in the scene 'scene_name'
                              Each entry in the array consists of (timestamp, rad_id, frame_id, closest_imu_meas_idx).
                              These are sorted entries in the increasing order of the timestamps
            : ego_summary - A numpy array of shape ( m, k ), where m is the length of the sequence of the ego measurement 
                            corrosponding to the scene 'scene_name'. Each entry in the array consists of (timestamp, ...).
                            these are sorted entries in the increasing order of the timestamps
    """
    data_dir = os.path.join(config.root_dir_rad, scene_name)
    can_data, can_attr = load_can_signal(data_dir)
    _, timestamps_meta_info, _ = scene_rad_meta_info(config.root_dir_rad, scene_name)

    # create radar frame info summary
    rad_timestamps = []
    rad_ids = []
    frame_ids = []
    for sensor in const.radar_location_attr:
        #timestamps = extract_single_radar_timestamp(sensor + '_rad', timestamps_meta_info)
        timestamps = extract_timestamp_single_sensor(sensor , timestamps_meta_info)
        rad_timestamps.append(timestamps)
        rad_ids.append(np.repeat(const.radar_id[sensor], timestamps.shape[0]))
        frame_ids.append(np.arange(1, timestamps.shape[0] + 1))

    rad_timestamps = np.concatenate(rad_timestamps, axis=0)
    rad_ids = np.concatenate(rad_ids, axis=0)
    frame_ids = np.concatenate(frame_ids, axis=0)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # create imu pr pose summary
    if sync_type == 'imu': ego_summary = create_imu_summary(can_data, can_attr)
    else: ego_summary = create_pose_summary(can_data, can_attr)

    # sort the radar frames by time
    frame_summary = frame_summary[np.argsort(frame_summary[:, 0])]

    # find the imu index temporally closest to a radar frame idx
    timestamps_radars =  np.expand_dims(frame_summary[:, 0], axis=-1)
    timestamps_ego = np.expand_dims(ego_summary[:, 0], axis=0)
    dt = np.abs(timestamps_radars - timestamps_ego)
    closest_ego_meas = np.argmin(dt, axis=1)

    # frame_summary
    closest_ego_meas = np.expand_dims(closest_ego_meas, axis=-1)
    frame_summary = np.concatenate([frame_summary, closest_ego_meas], axis=-1)
    return frame_summary, ego_summary

# =====================================================================================================================

def sync_cam_all_radars(scene_name):
    """ synchronize cameras with all the radars by associating each of the radar frames with the temporally closest camera frame
    inputs : scene_name - strings
    outputs : frame_summary - A numpy array of shape ( n, 4 ), where n is total number of frames of all the radars in the scene 'scene_name'
                              Each entry in the array consists of (timestamp, rad_id, frame_id).
                              These are sorted entries in the increasing order of the timestamps
            : cam_summary - A list of length (num_cam, ). Each entry of the list is a numpy array of shape ( n, 2 ), 
                            where n is total number of frames of all the radars in the scene 'scene_name' 
                            Each entry in the array consists of (timestamp, cam_frame_number).
    """
    data_dir = os.path.join(config.root_dir_rad, scene_name)
    can_data, can_attr = load_can_signal(data_dir)
    _, timestamps_meta_info, _ = scene_rad_meta_info(config.root_dir_rad, scene_name)
    _, _, _, cam_timestamps_meta_info = scene_cam_meta_info(config.root_dir_cam, scene_name)
    cam_timestamps = extract_timestamp_all_sensors(const.camera_location_attr, cam_timestamps_meta_info)

    # create radar frame info summary
    rad_timestamps = []
    rad_ids = []
    frame_ids = []
    for sensor in const.radar_location_attr:
        timestamps = extract_timestamp_single_sensor(sensor , timestamps_meta_info)
        rad_timestamps.append(timestamps)
        rad_ids.append(np.repeat(const.radar_id[sensor], timestamps.shape[0]))
        frame_ids.append(np.arange(1, timestamps.shape[0] + 1))

    rad_timestamps = np.concatenate(rad_timestamps, axis=0)
    rad_ids = np.concatenate(rad_ids, axis=0)
    frame_ids = np.concatenate(frame_ids, axis=0)
    frame_summary = np.stack([rad_timestamps, rad_ids, frame_ids], axis=-1)

    # sort the radar frames by time
    frame_summary = frame_summary[np.argsort(frame_summary[:, 0])]

    # closest camera info
    closest_cam_frame_id = []
    closest_cam_timestamps = []
    cam_summary = []

    # find the camera index temporally closest to a radar frame idx
    for cam in const.camera_location_attr:
        timestamps_rad = np.expand_dims(frame_summary[:, 0], axis=-1)
        timestamps_cam = np.expand_dims(cam_timestamps[cam], axis=0)
        dt = np.abs(timestamps_rad - timestamps_cam)
        closest_idx = np.argmin(dt, axis=1)
        closest_cam_frame_id.append(closest_idx)
        closest_cam_timestamps.append(cam_timestamps[cam][closest_idx])
        # print(closest_idx[:280])
        # input()

    # create the sync summary
    for i in range(const.num_cameras):
        timestamp = closest_cam_timestamps[i]
        closest_id = closest_cam_frame_id[i]
        summary = np.stack([timestamp, closest_id], axis=-1)
        cam_summary.append(summary)

    return frame_summary, cam_summary

# =====================================================================================================================







if __name__ == '__main__':

    scene_name = 'scene-0103'

    frame_summary1, imu_summary1 = sync_imu_single_radar(scene_name, 'front')
    frame_summary2, imu_summary2 = sync_imu_all_radars(scene_name)
    frame_summary3, pose_summary3 = sync_ego_sensor_all_attr_single_radar(scene_name, 'front', 'pose')
    frame_summary4, imu_summary4 = sync_ego_sensor_all_attr_all_radars(scene_name, 'imu')

    # print(frame_summary1)
    # print(imu_summary1)

    # print(frame_summary2)
    # print(imu_summary2)

    # print(frame_summary3)
    # print(pose_summary3)

    print(frame_summary4)
    print(imu_summary4)

