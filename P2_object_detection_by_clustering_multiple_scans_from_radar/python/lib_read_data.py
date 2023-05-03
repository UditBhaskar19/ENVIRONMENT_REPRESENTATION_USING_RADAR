# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : File reading utilities

# function names
# - parse_txt_file() : read a txt file line by line 
# - parse_csv_file() : read a csv file into a numpy array

# - load_can_signal() : load the gt pose data, vehicle sensor data, imu data corrosponding to a scene
# - scene_rad_meta_info() : create path for reading the radar data given the root directory and the scene name
# - scene_cam_meta_info() : create path for reading the cam data given the root directory and the scene name

# - extract_timestamp_single_sensor() : given the sensor name and timestamps info dictionary extract the sensor time stamps
# - extract_timestamp_all_sensors() : extract the timestamps of all sensors

# - extract_mounting_param_single_radar() : extract mount data of a single
# - extract_all_mounting_param_single_radar() : extract all the mount data of a single
# - extract_mounting_param_all_radars() : extract mounting parameters from all radars
# - extract_SE3_extrinsic_matrix_all_radars() : extract coordinate transformation matrix ( SE3 ) of all radars
# - extract_calib_matrix_single_camera() : extract the camera calibrartion matrix - both intrinsic and extrinsic 
# - extract_camera_calib_info() : extract the camera calibrartion matrix - both intrinsic and extrinsic for all the cameras

# - create_frames_list_single_radar() : create a list of radar frame directories for a single radar 
# - create_frames_list_all_radars() : create a list of radar frame directories for all radars
# - create_frame_list_single_camera() : create a list of camera image directories for a single camera
# - create_frame_list_all_cameras() : create a list of camera image directories for a all camera 
# - get_camera_frame() : use opencv to read an image

# measurement selction function :
# - select_stationary_high_confidence_meas_radar() : select stationary high confident measurements
# - select_moving_high_confidence_meas_radar() : select moving high confident measurements
# - select_stationary_all_valid_meas_radar() : select stationary all valid measurements
# - select_moving_all_valid_meas_radar() : select moving all valid measurements
# - select_stationary_all_meas_radar() : select stationary all measurements
# - select_moving_all_meas_radar() : select moving all measurements
# - select_all_invalid_meas_radar() : select moving all invalid measurements
# - select_all_valid_meas_radar() : select all valid measurements
# - select_all_meas_radar() : select all measurements
# =====================================================================================================================

import const, config
import csv, os
import numpy as np
from lib_functions import convert_quaternion2eular_angles, radar_SE3_element

# =====================================================================================================================

def parse_txt_file(file): 
    """ read a txt file line by line  """
    lines = []
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            lines.append(line)
    return lines

# =====================================================================================================================

def parse_csv_file(file, header=True):
    """ read a csv file into a numpy array """
    data = []
    fields = None
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)      
        if header:
            fields = next(csvreader)             
        for row in csvreader:                
            data.append(row)
        data = np.array(data, dtype = np.float32)
    return fields, data

# =====================================================================================================================

def load_can_signal(scene_dir):
    """ load all the sensor data except from the radar 
    (gt pose data, vehicle sensor data, imu data) corrosponding to a scene
    inputs: scene_dir - string
    outputs: signal_attr - a dictionary of sensor data attribute names and csv column idx
           : signal - a dictionary of sensor data
    """
    file = os.path.join(scene_dir, 'can', 'gt_pose.csv')
    pose_fields, pose_data = parse_csv_file(file)
    pose_fields = {pose_fields[i]:i for i in range(len(pose_fields))}

    file = os.path.join(scene_dir, 'can', 'zoe_veh_signal.csv')
    zoe_veh_fields, zoe_veh_data = parse_csv_file(file)
    zoe_veh_fields = {zoe_veh_fields[i]:i for i in range(len(zoe_veh_fields))}

    file = os.path.join(scene_dir, 'can', 'imu_signal.csv')
    imu_fields, imu_data = parse_csv_file(file)
    imu_fields = {imu_fields[i]:i for i in range(len(imu_fields))}
    
    signal_attr = {
        'pose_fields': pose_fields,
        'zoe_veh_fields': zoe_veh_fields,
        'imu_fields': imu_fields
    }
    signal = {
        'pose_data': pose_data, 
        'zoe_veh_data': zoe_veh_data,
        'imu_data': imu_data,
    }
    return signal, signal_attr

# =====================================================================================================================

def scene_rad_meta_info(root_dir, scene_name):
    """ create path for reading the radar data given the root directory and the scene name, 
    the radar data consists of radar frames directory, radar time stamps csv and radar mount parameters  
    inputs : root_dir - string 
           : scene_name - string
    outputs : rad_data_meta_info - a dictionary of radar frame directiories
            : timestamps_meta_info - a dictionary of radar timestamps csv file paths
            : calib_meta_info - a dictionary of radar calibration file paths
    """
    scene_dir = os.path.join(root_dir, scene_name)

    front_rad_data_dir = os.path.join(scene_dir, 'radar', const.radar_location_attr[0])
    left_rad_data_dir = os.path.join(scene_dir, 'radar', const.radar_location_attr[1])
    rear_left_rad_data_dir = os.path.join(scene_dir, 'radar', const.radar_location_attr[2])
    rear_right_rad_data_dir = os.path.join(scene_dir, 'radar', const.radar_location_attr[3])
    right_rad_data_dir = os.path.join(scene_dir, 'radar', const.radar_location_attr[4])

    front_rad_time_stamps = os.path.join(scene_dir, 'radar', const.radar_location_attr[0], 'time_stamps_sec.csv')
    left_rad_time_stamps = os.path.join(scene_dir, 'radar', const.radar_location_attr[1], 'time_stamps_sec.csv')
    rear_left_rad_time_stamps = os.path.join(scene_dir, 'radar', const.radar_location_attr[2], 'time_stamps_sec.csv')
    rear_right_rad_time_stamps = os.path.join(scene_dir, 'radar', const.radar_location_attr[3], 'time_stamps_sec.csv')
    right_rad_time_stamps = os.path.join(scene_dir, 'radar', const.radar_location_attr[4], 'time_stamps_sec.csv')

    front_rad_calib_file = os.path.join(scene_dir, 'radar', const.radar_location_attr[0] + '_calib_extrinsic.csv')
    left_rad_calib_file = os.path.join(scene_dir, 'radar', const.radar_location_attr[1] + '_calib_extrinsic.csv')
    rear_left_rad_calib_file = os.path.join(scene_dir, 'radar', const.radar_location_attr[2] + '_calib_extrinsic.csv')
    rear_right_rad_calib_file = os.path.join(scene_dir, 'radar', const.radar_location_attr[3] + '_calib_extrinsic.csv')
    right_rad_calib_file = os.path.join(scene_dir, 'radar', const.radar_location_attr[4] + '_calib_extrinsic.csv')

    rad_data_meta_info = {
        'front_data_dir': front_rad_data_dir,
        'left_data_dir': left_rad_data_dir,
        'rear_left_data_dir': rear_left_rad_data_dir,
        'rear_right_data_dir': rear_right_rad_data_dir,
        'right_data_dir': right_rad_data_dir
    }

    rad_timestamps_meta_info = {
        'front_time_stamps': front_rad_time_stamps,
        'left_time_stamps': left_rad_time_stamps,
        'rear_left_time_stamps': rear_left_rad_time_stamps,
        'rear_right_time_stamps': rear_right_rad_time_stamps,
        'right_time_stamps': right_rad_time_stamps
    }

    rad_calib_meta_info = {
        'front_calib_file': front_rad_calib_file,
        'left_calib_file': left_rad_calib_file,
        'rear_left_calib_file': rear_left_rad_calib_file,
        'rear_right_calib_file': rear_right_rad_calib_file,
        'right_calib_file': right_rad_calib_file
    }

    return (
        rad_data_meta_info, 
        rad_timestamps_meta_info, 
        rad_calib_meta_info)

# =====================================================================================================================

def scene_cam_meta_info(root_dir, scene_name):
    """ create path for reading the camera data given the root directory and the scene name, 
    the camera data consists of frames directory, time stamps csv and mount parameters  
    inputs : root_dir - string 
           : scene_name - string
    outputs : cam_intrinsic_data_meta_info
            : cam_extrinsic_data_meta_info
            : cam_frame_names
            : cam_time_stamps_sec
    """
    cam_frame_info_dir = os.path.join(root_dir, 'frame_info', scene_name, 'camera')

    # camera extrinsic param
    front_calib_extrinsic = os.path.join(cam_frame_info_dir, 'front_calib_extrinsic.csv')
    front_left_calib_extrinsic = os.path.join(cam_frame_info_dir, 'front_left_calib_extrinsic.csv')
    front_right_calib_extrinsic = os.path.join(cam_frame_info_dir, 'front_right_calib_extrinsic.csv')
    rear_calib_extrinsic = os.path.join(cam_frame_info_dir, 'rear_calib_extrinsic.csv')
    rear_left_calib_extrinsic = os.path.join(cam_frame_info_dir, 'rear_left_calib_extrinsic.csv')
    rear_right_calib_extrinsic = os.path.join(cam_frame_info_dir, 'rear_right_calib_extrinsic.csv')

    # camera intrinsic param
    front_calib_intrinsic = os.path.join(cam_frame_info_dir, 'front_calib_intrinsic.csv')
    front_left_calib_intrinsic = os.path.join(cam_frame_info_dir, 'front_left_calib_intrinsic.csv')
    front_right_calib_intrinsic = os.path.join(cam_frame_info_dir, 'front_right_calib_intrinsic.csv')
    rear_calib_intrinsic = os.path.join(cam_frame_info_dir, 'rear_calib_intrinsic.csv')
    rear_left_calib_intrinsic = os.path.join(cam_frame_info_dir, 'rear_left_calib_intrinsic.csv')
    rear_right_calib_intrinsic = os.path.join(cam_frame_info_dir, 'rear_right_calib_intrinsic.csv')

    # camera frame names
    front_cam_frames = os.path.join(cam_frame_info_dir, 'front_cam_frames.txt')
    front_left_cam_frames = os.path.join(cam_frame_info_dir, 'front_left_cam_frames.txt')
    front_right_cam_frames = os.path.join(cam_frame_info_dir, 'front_right_cam_frames.txt')
    rear_cam_frames = os.path.join(cam_frame_info_dir, 'rear_cam_frames.txt')
    rear_left_cam_frames = os.path.join(cam_frame_info_dir, 'rear_left_cam_frames.txt')
    rear_right_cam_frame = os.path.join(cam_frame_info_dir, 'rear_right_cam_frames.txt')

    # camera timestamps
    front_time_stamps_sec = os.path.join(cam_frame_info_dir, 'front_time_stamps_sec.csv')
    front_left_time_stamps_sec = os.path.join(cam_frame_info_dir, 'front_left_time_stamps_sec.csv')
    front_right_time_stamps_sec = os.path.join(cam_frame_info_dir, 'front_right_time_stamps_sec.csv')
    rear_left_time_stamps_sec = os.path.join(cam_frame_info_dir, 'rear_left_time_stamps_sec.csv')
    rear_right_time_stamps_sec = os.path.join(cam_frame_info_dir, 'rear_right_time_stamps_sec.csv')
    rear_time_stamps_sec = os.path.join(cam_frame_info_dir, 'rear_time_stamps_sec.csv')

    cam_intrinsic_data_meta_info = {
        'front_calib_intrinsic': front_calib_intrinsic,
        'front_left_calib_intrinsic': front_left_calib_intrinsic,
        'front_right_calib_intrinsic': front_right_calib_intrinsic,
        'rear_calib_intrinsic': rear_calib_intrinsic,
        'rear_left_calib_intrinsic': rear_left_calib_intrinsic,
        'rear_right_calib_intrinsic': rear_right_calib_intrinsic
    }

    cam_extrinsic_data_meta_info = {
        'front_calib_extrinsic': front_calib_extrinsic,
        'front_left_calib_extrinsic': front_left_calib_extrinsic,
        'front_right_calib_extrinsic': front_right_calib_extrinsic,
        'rear_calib_extrinsic': rear_calib_extrinsic,
        'rear_left_calib_extrinsic': rear_left_calib_extrinsic,
        'rear_right_calib_extrinsic': rear_right_calib_extrinsic
    }

    cam_frame_names = {
        'front_frames': front_cam_frames,
        'front_left_frames': front_left_cam_frames,
        'front_right_frames': front_right_cam_frames,
        'rear_frames': rear_cam_frames,
        'rear_left_frames': rear_left_cam_frames,
        'rear_right_frames': rear_right_cam_frame
    }

    cam_time_stamps_sec = {
        'front_time_stamps': front_time_stamps_sec,
        'front_left_time_stamps': front_left_time_stamps_sec,
        'front_right_time_stamps': front_right_time_stamps_sec,
        'rear_left_time_stamps': rear_left_time_stamps_sec,
        'rear_right_time_stamps': rear_right_time_stamps_sec,
        'rear_time_stamps': rear_time_stamps_sec
    }

    return (
        cam_intrinsic_data_meta_info, 
        cam_extrinsic_data_meta_info, 
        cam_frame_names, 
        cam_time_stamps_sec)

# =====================================================================================================================

def extract_timestamp_single_sensor(sensor, timestamps_meta_info):
    """ given the sensor name and timestamps info dictionary extract the sensor time stamps
    inputs: sensor - sensor name
          : timestamps_meta_info - a dictionary of timestamps csv file paths
    outputs : timestamp - timestamps as numpy array of shape (n, )
    """
    _, timestamp = parse_csv_file(timestamps_meta_info[sensor + '_time_stamps'])
    timestamp = timestamp.reshape(-1)
    return timestamp

# =====================================================================================================================

def extract_timestamp_all_sensors(sensor_locations, timestamps_meta_info):
    """ given a list of sensor names and timestamps info dictionary extract the sensor timestamps
    inputs: sensor_locations - list of sensor locations
          : timestamps_meta_info - a dictionary of timestamps csv file paths
    outputs : timestamp - timestamps as numpy array of shape (n, )
    """
    timestamp = {}
    for sensor in sensor_locations:
        _, timestamp_single_snsr = parse_csv_file(timestamps_meta_info[sensor + '_time_stamps'])
        timestamp[sensor] = timestamp_single_snsr.reshape(-1)
    return timestamp

# =====================================================================================================================

def extract_mounting_param_single_radar(sensor, calib_meta_info):
    """ given the radar name and calibration info dictionary extract the sensor mount data 
    inputs: sensor - radar name 
          : calib_meta_info - a dictionary of radar calibration file paths
    outputs: rad_mount_x - radar mount x coordinate
           : rad_mount_y - radar mount y coordinate
           : rad_mount_yaw - radar mount yaw 
    """
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

def extract_all_mounting_param_single_radar(sensor, calib_meta_info):
    """ given the radar name and calibration info dictionary extract the sensor mount data 
    inputs: sensor - radar name 
          : calib_meta_info - a dictionary of radar calibration file paths
    outputs: rad_mount_x - radar mount x coordinate
           : rad_mount_y - radar mount y coordinate
           : rad_mount_z - radar mount z coordinate
           : rad_mount_roll - radar mount roll 
           : rad_mount_pitch - radar mount pitch 
           : rad_mount_yaw - radar mount yaw 
    """
    calib_attr, rad_calib = parse_csv_file(calib_meta_info[sensor + '_calib_file'])
    rad_calib = rad_calib.reshape(-1)
    calib_attr = {calib_attr[i]:i for i in range(len(calib_attr))}
    Q = np.array([rad_calib[calib_attr['q1']], \
                  rad_calib[calib_attr['q2']], \
                  rad_calib[calib_attr['q3']], \
                  rad_calib[calib_attr['q4']]])
    rad_mount_roll, rad_mount_pitch, rad_mount_yaw = convert_quaternion2eular_angles(Q)
    rad_mount_x = rad_calib[calib_attr['Tx']]
    rad_mount_y = rad_calib[calib_attr['Ty']]
    rad_mount_z = rad_calib[calib_attr['Tz']]
    return (
        rad_mount_x, 
        rad_mount_y, 
        rad_mount_z,
        rad_mount_roll, 
        rad_mount_pitch,
        rad_mount_yaw,
    )

# =====================================================================================================================

def extract_mounting_param_all_radars(radar_locations, calib_meta_info):
    """ extract mounting parameters from all radars
    inputs: scene_name - string
    outputs: mount_parameters -  a numpy array of radar mount parameters of shape (num radars, 3)
    """
    mount_param = []
    for radar in radar_locations:
        mount_x, mount_y, mount_yaw = extract_mounting_param_single_radar(radar, calib_meta_info)
        mount_param.append(np.array([mount_x, mount_y, mount_yaw], dtype=np.float32))
    mount_param = np.stack(mount_param, axis=0)
    return mount_param

# =====================================================================================================================

def extract_radar_mount_info(scene_name):
    """ extract mounting parameters from all radars
    inputs: scene_name - string
    outputs: mount_parameters -  a numpy array of radar mount parameters of shape (num radars, 3)
    """
    mount_parameters = []
    _, _, calib_meta_info = scene_rad_meta_info(config.root_dir_rad, scene_name)
    for i in range(const.num_radars):
        radar = const.radar_location_attr[i]
        mount_x, mount_y, mount_yaw = extract_mounting_param_single_radar(radar, calib_meta_info)
        mount_parameters.append(np.array([mount_x, mount_y, mount_yaw], dtype=np.float32))
    mount_parameters = np.stack(mount_parameters, axis=0)
    return mount_parameters

# =====================================================================================================================

def extract_SE3_extrinsic_matrix_all_radars(radar_locations, calib_meta_info):
    """ extract coordinate transformation matrix ( SE3 ) of all radars
    inputs: scene_name - string
    outputs: mount_parameters -  a numpy array of radar mount parameters of shape (num radars, 3)
    """
    mount_param = []
    for radar in radar_locations:
        mount_x, mount_y, mount_z, mount_roll, mount_pitch, mount_yaw = extract_all_mounting_param_single_radar(radar, calib_meta_info)
        mount_param.append(radar_SE3_element(mount_x, mount_y, mount_z, mount_roll, mount_pitch, mount_yaw))
    mount_param = np.stack(mount_param, axis=0)
    return mount_param

# =====================================================================================================================

def extract_calib_matrix_single_camera(camera, cam_extrinsic_data_meta_info, cam_intrinsic_data_meta_info):
    """ extract the camera calibrartion matrix - both intrinsic and extrinsic 
    inputs: camera - camera name ( string )
          : cam_extrinsic_data_meta_info 
          : cam_intrinsic_data_meta_info
    outputs: cam_extrinsic_matrix, cam_intrinsic_matrix - numpy arrays of intrinsic and extrinsic camera matrix
    """
    _, cam_extrinsic_matrix = parse_csv_file(cam_extrinsic_data_meta_info[camera + '_calib_extrinsic'], header=False)
    _, cam_intrinsic_matrix = parse_csv_file(cam_intrinsic_data_meta_info[camera + '_calib_intrinsic'], header=False)
    return cam_extrinsic_matrix, cam_intrinsic_matrix

# =====================================================================================================================

def extract_camera_calib_info(camera_locations, cam_extrinsic_data_meta_info, cam_intrinsic_data_meta_info):
    """ extract the camera calibrartion matrix - both intrinsic and extrinsic for all the cameras
    inputs: camera_locations - a list of camera names
          : cam_extrinsic_data_meta_info 
          : cam_intrinsic_data_meta_info
    outputs: cam_extrinsic_matrix, cam_intrinsic_matrix - numpy arrays of intrinsic and extrinsic camera matrix for all cameras
    """
    extrinsic_calib_matrix = []
    intrinsic_calib_matrix = []
    for camera in camera_locations:
        extrinsic, intrinsic \
            = extract_calib_matrix_single_camera(camera, cam_extrinsic_data_meta_info, cam_intrinsic_data_meta_info)
        extrinsic_calib_matrix.append(extrinsic)
        intrinsic_calib_matrix.append(intrinsic)
    extrinsic_calib_matrix = np.stack(extrinsic_calib_matrix, axis=0)
    intrinsic_calib_matrix = np.stack(intrinsic_calib_matrix, axis=0)
    return extrinsic_calib_matrix, intrinsic_calib_matrix

# =====================================================================================================================

def create_frames_list_single_radar(sensor, num_frames, rad_data_meta_info):
    """ create a list of radar frame directories for a single radar 
    inputs: sensor - radar name
          : num_frames - number of frames for a scene and a radar
          : rad_data_meta_info - 
    outputs: files -  a list of radar frame directories
    """
    files = []
    offset = 1
    for idx in range(offset, num_frames + offset):
        files.append(os.path.join(rad_data_meta_info[sensor + '_data_dir'], str(idx) + '.csv'))
    return files

# =====================================================================================================================

def create_frames_list_all_radars(radar_locations, num_frames, rad_data_meta_info):
    """ create a list of radar frame directories for all radars
    inputs: radar_locations - a list of radar locvations
          : num_frames - number of frames for a scene and a radar
          : rad_data_meta_info - 
    outputs: files -  a list of list of radar frame directories
    """
    files = {}
    for i in range(len(radar_locations)):
        files[radar_locations[i]] = create_frames_list_single_radar(radar_locations[i], num_frames[i], rad_data_meta_info)
    return files

# =====================================================================================================================

def create_frame_list_single_camera(sensor, cam_frame_names_meta_info):
    """ create a list of camera image directories for a single camera 
    inputs: sensor - camera name
          : cam_frame_names_meta_info - 
    outputs: files -  a list of camera image directories
    """
    files = parse_txt_file(cam_frame_names_meta_info[sensor + '_frames'])
    file_names = [ os.path.join(config.root_dir_cam, file ) for file in files]
    return file_names

# =====================================================================================================================

def create_frame_list_all_cameras(camera_locations, cam_frame_names_meta_info):
    """ create a list of camera image directories for a all camera 
    inputs: camera_locations - a list of camera names
          : cam_frame_names_meta_info - 
    outputs: files -  a list of list of camera image directories
    """
    files = {}
    for cam in camera_locations:
        frames = create_frame_list_single_camera(cam, cam_frame_names_meta_info)
        files[cam] = frames
    return files

# =====================================================================================================================

def get_camera_frame(image_path):
    """ use opencv to read an image """
    return cv2.imread(image_path)

# =====================================================================================================================

def extract_radar_frame(
    file, 
    measurement_selection_function):
    """ load radar frame from csv file
    inputs: file - radar csv file path
          : measurement_selection_function - a function that selects measurements as per specific conditions
    outputs: z - numpy array of radar measurements (px, py, vx, vy) of shape (m, 4)
           : z_rms - numpy array of radar measurement rms (x_rms, y_rms, vx_rms, vy_rms) of shape (m, 4)
           : phd0 - numpy array of probability that the measurement is valid of shape (m, )
    """
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

def select_stationary_high_confidence_meas_radar(
    radar_meas, 
    radar_attr):
    """ measurement selection function : select stationary high confident measurements """
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
    """ measurement selection function : select moving high confident measurements """
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
    """ measurement selction function : select stationary all valid measurements """
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
    """ measurement selction function : select moving all valid measurements """
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

def select_stationary_all_meas_radar(
    radar_meas, 
    radar_attr):
    """ measurement selction function : select stationary all valid measurements """
    dyn_prop_vals = radar_meas[:, radar_attr['dyn_prop']]
    condition = np.logical_or(              # stationary measurements only
        np.int16(dyn_prop_vals)==1 , 
        np.int16(dyn_prop_vals)==3 , 
        np.int16(dyn_prop_vals)==5
    )
    return radar_meas[condition, :]

# =====================================================================================================================

def select_moving_all_meas_radar(
    radar_meas, 
    radar_attr):
    """ measurement selction function : select moving all measurements """
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
    """ measurement selction function : select all invalid measurements """
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

def select_all_valid_meas_radar(
    radar_meas, 
    radar_attr):
    """ measurement selction function : select all valid measurements """
    valid_meas_vals = radar_meas[:, radar_attr['valid_state']]
    condition = np.zeros((valid_meas_vals.shape[0], ), dtype=np.bool8)
    is_equal_vals = [0,4,8,9,10,11,12,15,16,17]
    for val in is_equal_vals:
        condition = np.logical_or(condition, np.int16(valid_meas_vals)==val)
    return radar_meas[condition, :]

# =====================================================================================================================

def select_all_meas_radar(
    radar_meas, 
    radar_attr):
    """ measurement selction function : select all measurements """
    return radar_meas

# =====================================================================================================================






if __name__ == '__main__':

    scene = 'scene-1094'          #scene-0655, scene-0103, scene-0796, scene-1077, scene-1094

    # extract the meta info
    rad_data_meta_info, rad_timestamps_meta_info, rad_calib_meta_info \
        = scene_rad_meta_info(config.root_dir_rad, scene)
    cam_intrinsic_data_meta_info, cam_extrinsic_data_meta_info, cam_frame_names_meta_info, cam_timestamps_meta_info \
        = scene_cam_meta_info(config.root_dir_cam, scene)

    # extract timestamp
    rad_timestamps = extract_timestamp_all_sensors(const.radar_location_attr, rad_timestamps_meta_info)
    cam_timestamps = extract_timestamp_all_sensors(const.camera_location_attr, cam_timestamps_meta_info)

    # extracting mounting info
    extrinsic_calib_matrix_cam, intrinsic_calib_matrix_cam \
        = extract_camera_calib_info(const.camera_location_attr, cam_extrinsic_data_meta_info, cam_intrinsic_data_meta_info)
    extrinsic_calib_matrix_rad = extract_SE3_extrinsic_matrix_all_radars(const.radar_location_attr, rad_calib_meta_info)
    mounting_param = extract_mounting_param_all_radars(const.radar_location_attr, rad_calib_meta_info)

    # number of frames radars
    num_frames = [ len(rad_timestamps[rad]) for rad in const.radar_location_attr ]

    # extract frame list
    radar_frames = create_frames_list_all_radars(const.radar_location_attr, num_frames, rad_data_meta_info)
    camera_frames = create_frame_list_all_cameras(const.camera_location_attr, cam_frame_names_meta_info)

    # =====================================================================================================================
    for rad in const.radar_location_attr:
        print(rad_timestamps[rad].shape)
    print('-'*30)
    for cam in const.camera_location_attr:
        print(cam_timestamps[cam].shape)
    # =====================================================================================================================
    print(extrinsic_calib_matrix_cam.shape)
    print(intrinsic_calib_matrix_cam.shape)
    print(extrinsic_calib_matrix_rad.shape)
    print(mounting_param.shape)
    print('-'*30)
    # =====================================================================================================================
    print(num_frames)
    print('-'*30)
    # =====================================================================================================================
    # for rad in const.radar_location_attr:
    #     print(radar_frames[rad])
    #     print('-'*30)
    # =====================================================================================================================
    # for cam in const.camera_location_attr:
    #     print(camera_frames[cam])
    #     print('-'*30)
    # =====================================================================================================================
    for rad in const.radar_location_attr:
        print(rad)
        for path in radar_frames[rad]:
            z, z_rms, pdh0 = extract_radar_frame(path, select_moving_all_valid_meas_radar)
        print('-'*30)

    print(const.radar_id)