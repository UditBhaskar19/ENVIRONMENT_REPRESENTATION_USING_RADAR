# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library for ego compensation
# =====================================================================================================================

import numpy as np

def construct_SE2_group_element(px, py, theta):
    """ given a pose construct a trasformation matrix T """
    T = np.eye(3)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = np.array([[px],[py]])
    T[:2, :2] = R
    T[:2, 2:] = t
    return T


def inverse_SE2(T):
    """ the inverse of the transformation matrix T """
    R = T[:2, :2]
    t = T[:2, 2:]
    T = np.eye(3)
    T[:2, :2] = R.transpose()
    T[:2, 2:] = -R.transpose() @ t
    return T


def coordinate_transform_meas(meas_x, meas_y, T):
    """ coordinate transform measurements given the matrix T """
    R = T[:2, :2]
    t = T[:2, 2:]
    meas = np.stack([meas_x, meas_y], axis=0)
    meas = R @ meas + t
    return meas[0,:], meas[1,:]


def ego_compensate_prev_meas_sensor_frame(
    meas_x_prev, 
    meas_y_prev, 
    T_snsr, 
    T_pose_curr, 
    T_pose_prev):
    """ spatially align prev and curr sensor acans in the current sensor frame.
    Assuming that the measurements are in the sensor frame  """
    T =  np.linalg.inv( T_pose_curr @ T_snsr ) @ T_pose_prev @ T_snsr
    R = T[:2, :2]
    t = T[:2, 2:]
    meas_prev = np.stack([meas_x_prev, meas_y_prev], axis=0)
    meas_ego_comp = R @ meas_prev + t
    return meas_ego_comp[0,:], meas_ego_comp[1,:]


def ego_compensate_prev_meas_vehicle_frame(
    meas_x_prev, 
    meas_y_prev,
    T_pose_curr, 
    T_pose_prev):
    """ spatially align prev and curr sensor acans in the current ego-vehicle frame.
    Assuming that the measurements are in the ego vehicle frame  """
    T =  np.linalg.inv( T_pose_curr ) @ T_pose_prev
    R = T[:2, :2]
    t = T[:2, 2:]
    meas_prev = np.stack([meas_x_prev, meas_y_prev], axis=0)
    meas_ego_comp = R @ meas_prev + t
    return meas_ego_comp[0,:], meas_ego_comp[1,:]


def odom_step(px, py, yaw, vx, yaw_rate, dt):
    """ predict the ego vehicle pose after dt given the previous ego vehicle pose and ego motion
    Input: odom_prev_state - prev ego pose vector - (px, py, yaw)
         : ego_motion - ego motion vector - (vx, yaw_rate) - vy is assumed to be 0 
         : dt - time interval
    Output: odom_pred_state - predicted ego pose vector - (px, py, yaw)
    """
    vy = 0.0
    # some intermediate values
    dx = np.array([[vx*dt],[vy*dt]])
    dyaw = yaw_rate * dt
    sin_dtheta = np.sin(dyaw)
    cos_dtheta = np.cos(dyaw)

    x_prev = np.eye(3) 
    x_prev[:2, :2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw),  np.cos(yaw)]])
    x_prev[:2, -1] = np.array([px, py])

    # compute state transition matrix
    gamma0 = np.array([[cos_dtheta,  -sin_dtheta],  [sin_dtheta,  cos_dtheta]])
    if yaw_rate > 0.0:
        gamma1 = ( 1 / dyaw ) * ( np.array([[sin_dtheta,  cos_dtheta-1],  [1-cos_dtheta,  sin_dtheta]]) @ dx )
    else:
        gamma1 = np.eye(2) @ dx 
    phi = np.hstack((gamma0, gamma1))
    phi = np.vstack((phi, np.array([0, 0, 1])))
    
    # output
    x = x_prev @ phi
    px = x[0, -1]
    py = x[1, -1]
    yaw = yaw + dyaw
    return px, py, yaw


def integrate_ego_motion(px, py, yaw, vx, yaw_rate, dt):
    """ Given a sequence of ego motion and an initial pose, 
    compute the final ego pose """
    for t in range(dt.shape[0]):
        px, py, yaw = odom_step(px, py, yaw, vx[t], yaw_rate[t], dt[t])
    return px, py, yaw


def sync_radar_with_egomotion(meas_x, meas_y, t_meas, ego_vx, ego_yaw_rate, t_ego):
    """ temporally allign sensor measurements and the vehicle odometry """
    dt = np.abs(t_meas - t_ego)
    if dt == 0.0: return meas_x, meas_y
    px, py, yaw = odom_step(0.0, 0.0, 0.0, ego_vx, ego_yaw_rate, dt)
    T = construct_SE2_group_element(px, py, yaw)
    if t_ego < t_meas:
        meas_x, meas_y = coordinate_transform_meas(meas_x, meas_y, T)
    else:
        meas_x, meas_y = coordinate_transform_meas(meas_x, meas_y, inverse_SE2(T))
    return meas_x, meas_y


