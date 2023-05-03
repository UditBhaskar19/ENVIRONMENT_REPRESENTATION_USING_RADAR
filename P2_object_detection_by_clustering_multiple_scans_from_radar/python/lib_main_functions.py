# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : some common utilities used for coordinate transformation, radar frame accumulation, clustering etc

# function names
# - extract_sensor_data() : extract radar measurements, mount info and the synchronised imu meas from a sequence of radar frames
# - coordinate_transform() : coordinate transformation of measurements from sensor frame to vehicle frame
# - dynamic_meas_ego_compensation() : Ego motion compensation of the meas history buffer so that the past measurements are in the current ego-vehicle frame
# - ego_motion_compensate_measurements() : Ego motion compensation of the measurement history for all radars except the current active radar
# - accumulate_measurements() : Ego motion compensation of the measurement history for the current active radar
# - accumulate_radar_frames() : update the frame buffers 
# - perform_clustering() : perform clustering for a single radar
# =====================================================================================================================

import const, config
import numpy as np
from PIL import Image
from lib_read_data import extract_radar_frame
from lib_sampling import generate_radar_meas
from lib_clustering import compute_cov_ellipse
from lib_functions import (
    coordinate_transform_px_py_sf_to_vf,
    coordinate_transform_vx_vy_sf_to_vf,
    coordinate_transform_cov_sf_to_vf,
    coordinate_transform_px_py_vf_to_sf,
    dynamic_measurement_extrapolation,
    ego_compensate_meas_covariance,
    project_radar_on_camera
)

# ====================================================================================================================

def extract_sensor_data(frameid, conf):
    """ extract radar measurements, mount info and the synchronised imu meas from a sequence of radar frames
    inputs: frameid - time index
          : conf - configuration object. The below parameters are used from conf object
                        : meas_selection_function - function that selects a subset of radar measurements ( refer lib_read_data.py for more details )
                        : frame_summary -  radar frame summary for a scene ( refer lib_sync.py for more details )
                        : imu_summary - imu summary for a scene ( refer lib_sync.py for more details )
    Outputs : radar_id - radar name and radar id
            : radar_time_stamp - radar frame timestamp
            : rad_meas, rad_meas_rms, phd0 - radar meas, meas rms, probability that the meas is valid
            : imu_meas - synchronised imu meas
    """
    # get radar frame timestamp, radar id, frame_num and the radar name
    radar_time_stamp = conf.frame_summary[frameid, 0]
    radar_id = conf.frame_summary[frameid, 1].astype(np.int16)
    frame_id = conf.frame_summary[frameid, 2].astype(np.int16)
    radar_name = const.radar_id_to_name[radar_id]

    # get the radar data and  extract temporally closest imu data
    frame_dir = conf.radar_frames[radar_name][frame_id - 1]
    rad_meas, rad_meas_rms, phd0  = extract_radar_frame(frame_dir, conf.meas_selection_function)
    imu_meas = conf.imu_summary[conf.frame_summary[frameid, 3].astype(np.int16)]

    return ( 
        radar_id, radar_time_stamp, \
        rad_meas, rad_meas_rms, phd0, \
        imu_meas )

# ====================================================================================================================

def coordinate_transform(
    rad_meas, 
    meas_cov, 
    mount_tx, 
    mount_ty, 
    mount_yaw):
    """ coordinate transformation of measurements from sensor frame to vehicle frame
    Inputs : rad_meas - radar measurements array of shape (m, 4)
           : meas_cov - radar measurement covariances of shape (m, 4, 4)
           : mount_tx, mount_ty, mount_yaw - radar mount parameters
    Output : rad_meas_cts - coordinate transformed measurements
           : meas_cov_cts - coordinate transformed meas noise covariance
    """
    # coordinate transform measurements
    rad_meas_px, rad_meas_py = coordinate_transform_px_py_sf_to_vf(rad_meas[:, 0], rad_meas[:, 1], mount_tx, mount_ty, mount_yaw)
    rad_meas_vx, rad_meas_vy = coordinate_transform_vx_vy_sf_to_vf(rad_meas[:, 2], rad_meas[:, 3], mount_yaw)
    rad_meas_cts = np.stack([rad_meas_px, rad_meas_py, rad_meas_vx, rad_meas_vy], axis=-1)

    # coordinate transform measurement covariance
    meas_cov_cts = np.zeros((rad_meas_cts.shape[0], config.meas_dim, config.meas_dim), dtype=np.float32)
    meas_cov_cts[:, :2, :2] = coordinate_transform_cov_sf_to_vf(meas_cov[:, :2, :2], mount_yaw)
    meas_cov_cts[:, 2:, 2:] = coordinate_transform_cov_sf_to_vf(meas_cov[:, 2:, 2:], mount_yaw)

    return rad_meas_cts, meas_cov_cts

# ====================================================================================================================

def dynamic_meas_ego_compensation(meas_hist, ego_yaw_rate, dt):
    """ Ego motion compensation of the meas history buffer so that the past measurements are in the current ego-vehicle frame
    Inputs: meas_hist - data structure containing a history of radar frames ( refer lib_datastruct.py for more details )
          : ego_yaw_rate - yaw rate of the ego vehicle at the current time
          : dt - difference between current and previous time stamps
    Outputs: meas_hist - ego motion compensated meas history buffer
    """
    for i in range(meas_hist.get_buffer_size()):
        num_meas_i = meas_hist.get_num_meas_i(i)

        if num_meas_i > 0:
            meas_vector_i = meas_hist.get_meas_vector_i(i)
            meas_covariance_i = meas_hist.get_meas_covariance_i(i)
            meas_status_i = meas_hist.get_meas_status_i(i)

            idx = np.arange(num_meas_i, dtype=np.int32)
            dynamic_meas_idx = idx[meas_status_i[:, 0]]
            if dynamic_meas_idx.shape[0] > 0:
                dynamic_meas = meas_vector_i[dynamic_meas_idx]
                dynamic_meas_cov = meas_covariance_i[dynamic_meas_idx]

                meas_px, meas_py, meas_vx, meas_vy \
                    = dynamic_measurement_extrapolation(
                        dynamic_meas[:,0], dynamic_meas[:,1], dynamic_meas[:,2], dynamic_meas[:,3], 
                        ego_yaw_rate, dt)
                dynamic_meas_cov = ego_compensate_meas_covariance(dynamic_meas_cov, ego_yaw_rate, dt)

                meas_hist.meas_vector[i, dynamic_meas_idx] = np.stack([meas_px, meas_py, meas_vx, meas_vy], axis=-1)
                meas_hist.meas_covariance[i, dynamic_meas_idx] = dynamic_meas_cov

    return meas_hist

# ====================================================================================================================

def ego_motion_compensate_measurements(
    meas_hist,
    ego_yaw_rate, 
    trigger,
    dt):
    """ Ego motion compensation of the measurement history for all radars except the current active radar
    Inputs: meas_hist - a list of meas history buffer, where each entry corrosponds to a radar
            ego_yaw_rate - yaw rate of the ego vehicle at the current time
            trigger - an array of boolean flag of shise num_radars, indicating if a radar has generated measurements for the first time. 
                      if True, it means that the radar has not yet generated measurements and the measurement history buffer is empty
                      if False, it means that the radar has generated mesurements previously and the measurement history buffer is NOT empty
            dt - sample time
    Output: meas_hist - ego motion updated list of meas history buffer
    """
    for i in range(const.num_radars):
        if trigger[i] == False:
            meas_hist[i] = dynamic_meas_ego_compensation(meas_hist[i], ego_yaw_rate, dt)
    return meas_hist

# ====================================================================================================================

def accumulate_measurements(
    meas_hist,
    are_samples,
    radar_id,
    rad_meas,
    rad_meas_cov, 
    ego_yaw_rate,
    t):
    """ Ego motion compensation of the measurement history for the current active radar
    and frame accumulation of the current radar frame
    Inputs: meas_hist - measurement history buffer for the current active radar
            are_samples - an array of boolean flag of shape (n, 2). each entry corrosponds to (dynamic-status, is_generated_sample)
            radar_id - radar id of the current active radar
            rad_meas - current radar meas of shape (n, 4)
            rad_meas_cov - current radar meas covariance of shape (n, 4, 4)
            ego_yaw_rate - yaw rate of the ego vehicle
            t - current time stamp 
    Outputs : meas_hist - updated meas history buffer
    """
    # update the history buffer
    num_meas = rad_meas.shape[0]
    dyn_flag = np.ones((num_meas, ), dtype=np.bool8)
    meas_status = np.stack([dyn_flag, are_samples], axis=-1)

    # integrate frames
    meas_hist.update_buffer(
        rad_meas, rad_meas_cov, meas_status,
        num_meas, t, radar_id)

    return meas_hist

# ====================================================================================================================

def accumulate_radar_frames(
    time_idx,
    conf,
    meas_hist,
    trigger,
    timestamp_rad_prev):

    # extract sensor data
    radar_id, radar_time_stamp, rad_meas_sf, rad_meas_rms, phd0, imu_meas = extract_sensor_data(time_idx, conf)

    # extract the measurements
    rad_meas, meas_cov, are_samples = generate_radar_meas(rad_meas_sf, rad_meas_rms, config.sampling_covariance, config.num_samples)

    # coordinate transformation sf to vf
    rad_meas, meas_cov = coordinate_transform(rad_meas, meas_cov, conf.mount_param[radar_id, 0], conf.mount_param[radar_id, 1], conf.mount_param[radar_id, 2])

    # ego-compensate for the remaining radars
    meas_hist = ego_motion_compensate_measurements(meas_hist, imu_meas[1], trigger,  radar_time_stamp - timestamp_rad_prev)

    # update history for current radar
    meas_hist[radar_id] = accumulate_measurements(meas_hist[radar_id], are_samples, radar_id, rad_meas, meas_cov, imu_meas[1], radar_time_stamp)

    # update trigger
    trigger[radar_id] = False

    return meas_hist, trigger, radar_time_stamp, radar_id

# ====================================================================================================================

def perform_clustering(radar, trigger, meas_hist, meas_clusters):
    sensorid = const.radar_id[radar]
    meas_vector = 0
    if trigger[sensorid] == False:
        total_num_meas, meas_vector, meas_covariance, \
        meas_status, num_valid_meas, timestamp, \
        top_idx, sensorids = meas_hist[sensorid].get_all_valid_data()
        if meas_vector.shape[0] > 0:
            rad_meas_dyn_status = np.ones((meas_vector.shape[0], ), dtype=np.bool8)
            meas_clusters.dbscan(meas_vector, meas_covariance, rad_meas_dyn_status)
    return meas_clusters, meas_vector

# ====================================================================================================================

