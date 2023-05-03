# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : some common utilities used for coordinate transformation, radar frame accumulation, clustering etc

# function names
# - convert_quaternion2eular_angles() : convert quaternion to eular angles
# - radar_SE3_element() : construct a 3D rigid body transformation matrix (SE3)
# - construct_rotation_matrix() : construct a 2D rotation matrix (SO2)
# - construct_SE2_group_element() : given a pose construct a trasformation matrix T (SE2)
# - ego_compensate_prev_meas_vehicle_frame() : ego compensate the measurements
# - convert_meas_polar_to_cartesian() : convert meas polar to cartesian
# - convert_meas_cart_to_polar() : convert meas cartesian to polar
# - compute_range_rate() : Compute range rate from px, py, vx, vy
# - rotate_vectors() : perform rotation of a 2D-vector (x, y) by theta
# - coordinate_transform_px_py_sf_to_vf() : Coordinate transformation of the position vectors from sensor frame to vehicle frame
# - coordinate_transform_px_py_vf_to_sf() : Coordinate transformation of the velocity vectors from sensor frame to vehicle frame
# - coordinate_transform_vx_vy_sf_to_vf() : Coordinate transformation of the position vectors from vehicle frame to sensor frame
# - coordinate_transform_vx_vy_vf_to_sf() : coordinate transformation of the velocity vectors from vehicle frame to sensor frame
# - coordinate_transform_cov_sf_to_vf() : coordinate transform covariance from sensor frame to vehicle frame
# - rotate_cov_matrix() : rotate the covariance matrix by theta
# - dynamic_measurement_extrapolation() : dynamic measurement extrapolation
# - ego_compensate_meas_covariance() : ego compensate the meas covariance

# functions used for projecting radar to camera image
#  - create_cts_matrix() - transformation matrix from radar frame to camera frame
#  - represent_radar_homogeneous_3D() - Represent radar 2D position (px, py) measurement to 3D (px, py, pz)
#  - create_projection_transformation() - tranformation matrix from radar frame to camera image plane
#  - project_radar_on_camera() - project radar measurement to camera image
# =====================================================================================================================

import numpy as np

# =====================================================================================================================

def convert_quaternion2eular_angles(Q):
    """ inputs : Q - querternion vector
        outputs : roll, pitch, yaw 
    """
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

def radar_SE3_element(tx, ty, tz, roll, pitch, yaw):
    """ construct a 3D rigid body transformation matrix (SE3) """
    Rx = np.array([[1,           0,              0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                   [0,             1,              0],
                   [np.sin(pitch), 0,  np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3,  3] = np.array([tx, ty, tz])
    T[:3, :3] = Rz @ Ry @ Rx
    return T

# =====================================================================================================================

def construct_rotation_matrix(theta):
    """ construct a 2D rotation matrix (SO2) given an angle in radian """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R

# =====================================================================================================================

def construct_SE2_group_element(px, py, theta):
    """ given a pose construct a trasformation matrix T (SE2) """
    T = np.eye(3)
    R = construct_rotation_matrix(theta)
    t = np.array([[px],[py]])
    T[:2, :2] = R
    T[:2, 2:] = t
    return T

# =====================================================================================================================

def ego_compensate_prev_meas_vehicle_frame(
    meas_x_prev, 
    meas_y_prev,
    T_pose_curr, 
    T_pose_prev):
    """ spatially align prev sensor scan in the current ego-vehicle frame.
    Assuming that the measurements are in the ego vehicle frame  """
    T =  np.linalg.inv( T_pose_curr ) @ T_pose_prev
    R = T[:2, :2]
    t = T[:2, 2:]
    meas_prev = np.stack([meas_x_prev, meas_y_prev], axis=0)
    meas_ego_comp = R @ meas_prev + t
    return meas_ego_comp[0,:], meas_ego_comp[1,:]

# =====================================================================================================================

def convert_meas_polar_to_cartesian(range, azimuth):
    """ convert meas polar to cartesian """
    x = range * np.cos(azimuth)
    y = range * np.sin(azimuth)
    return x, y

# =====================================================================================================================

def convert_meas_cart_to_polar(xcoord, ycoord):
    """ convert meas cartesian to polar"""
    range_coord = np.sqrt( xcoord**2 + ycoord**2 )
    azi_coord = np.arctan2(ycoord, xcoord)
    return range_coord, azi_coord

# =====================================================================================================================

def compute_range_rate(px,py,vx,vy):
    """ Compute range rate from px, py, vx, vy """
    r = np.sqrt( px**2 + py**2 )
    vr = ( px / r ) * vx + ( py / r ) * vy
    return vr

# =====================================================================================================================

def rotate_vectors(x, y, theta):
    """ perform rotation of a 2D-vector (x, y) by theta """
    x_cts = x * np.cos(theta) - y * np.sin(theta)
    y_cts = x * np.sin(theta) + y * np.cos(theta)
    return x_cts, y_cts

# =====================================================================================================================

def coordinate_transform_px_py_sf_to_vf(px, py, tx, ty, theta):
    """ Coordinate transformation of the position vector from sensor frame to vehicle frame
    Input: ( px, py ) - position vector
         : ( tx, ty ) - sensor mount coordinates
         : theta - sensor mount azimuth angle
    Output: ( px_cts, py_cts ) - coordinate transformed position vector
    """
    px_cts, py_cts = rotate_vectors(px, py, theta)
    px_cts = px_cts + tx
    py_cts = py_cts + ty
    return px_cts, py_cts

# =====================================================================================================================    

def coordinate_transform_vx_vy_sf_to_vf(vx, vy, theta):
    """ Coordinate transformation of the velocity vector from sensor frame to vehicle frame
    Input: ( vx, vy ) - velocity vector
         : theta - sensor mount azimuth angle
    Output: ( vx_cts, vy_cts ) - coordinate transformed velocity vector
    """
    vx_cts, vy_cts = rotate_vectors(vx, vy, theta)
    return vx_cts, vy_cts

# =====================================================================================================================

def coordinate_transform_px_py_vf_to_sf(px, py, tx, ty, theta):
    """ coordinate transform measurements from vehicle frame to sensor frame """
    px = px - tx
    py = py - ty
    px_cts, py_cts = rotate_vectors(px, py, -theta)
    return px_cts, py_cts

# =====================================================================================================================

def coordinate_transform_vx_vy_vf_to_sf(vx, vy, theta):
    """ coordinate transform measurements from vehicle frame to sensor frame """
    vx_cts, vy_cts = rotate_vectors(vx, vy, -theta)
    return vx_cts, vy_cts

# =====================================================================================================================

def rotate_cov_matrix(cov, theta):
    """ rotate the covariance matrix by theta """
    R = np.expand_dims(construct_rotation_matrix(theta), axis=0)
    cov = R @ cov @ R.transpose(0,2,1)
    return cov

def coordinate_transform_cov_sf_to_vf(cov, theta):
    """ coordinate transform covariance from sensor frame to vehicle frame """
    cov = rotate_cov_matrix(cov, theta)
    return cov

# =====================================================================================================================

def dynamic_measurement_extrapolation(meas_px, meas_py, meas_vx, meas_vy, ego_yaw_rate, dt):
    """ dynamic measurement extrapolation """
    dyaw = ego_yaw_rate * dt
    meas_px, meas_py = rotate_vectors(meas_px, meas_py, -dyaw)
    meas_vx_pred, meas_vy_pred = rotate_vectors(meas_vx, meas_vy, -dyaw)
    meas_px_pred = meas_px + meas_vx_pred * dt
    meas_py_pred = meas_py + meas_vy_pred * dt
    return meas_px_pred, meas_py_pred, meas_vx_pred, meas_vy_pred

# =====================================================================================================================

def ego_compensate_meas_covariance(meas_cov, ego_yaw_rate, dt):
    """ ego compensate the meas covariance """
    dyaw = ego_yaw_rate * dt
    meas_cov_comp = np.zeros_like(meas_cov)
    meas_cov_comp[:, :2, :2] = rotate_cov_matrix(meas_cov[:, :2, :2], -dyaw)
    meas_cov_comp[:, 2:, 2:] = rotate_cov_matrix(meas_cov[:, 2:, 2:], -dyaw)
    return meas_cov_comp

# =====================================================================================================================

def create_cts_matrix(T_rad, T_cam_extrin):
    """ transformation matrix from radar frame to camera frame """
    return np.linalg.inv(T_cam_extrin) @ T_rad

# =====================================================================================================================

def represent_radar_homogeneous_3D(rad_meas):
    """ Represent radar 2D position (px, py) measurement to 3D (px, py, pz) """
    z = np.ones((rad_meas.shape[0], 4), dtype=np.float32)
    z[:, :2] = rad_meas[:, :2]
    z[:,  2] = 1.1
    return z

# =====================================================================================================================

def create_projection_transformation(T_rad, T_cam_extrin, T_cam_intrin):
    """ tranformation matrix from radar frame to camera image plane """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = T_cam_intrin
    T = T @ np.linalg.inv(T_cam_extrin) @ T_rad
    return T

# =====================================================================================================================

def project_radar_on_camera(T, rad_meas):
    """ project radar measurement to camera image """
    z = represent_radar_homogeneous_3D(rad_meas)
    z = z @ T.transpose()
    z = z[:, :2] / z[:, 2:3]
    return z

# =====================================================================================================================