# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Configuration class
# =====================================================================================================================
import numpy as np
import const, config
from lib_read_data import (
    scene_rad_meta_info,
    scene_cam_meta_info,
    extract_timestamp_all_sensors,
    extract_mounting_param_all_radars,
    extract_camera_calib_info,
    extract_SE3_extrinsic_matrix_all_radars,
    create_frames_list_all_radars,
    create_frame_list_all_cameras
)
from lib_sync import (
    sync_imu_all_radars,
    sync_cam_all_radars
)
from lib_functions import create_projection_transformation
from lib_grid import FOV_grid_coverage



class Config:

    def __init__(self, scene, meas_selection_function, buffer_size):

        self.scene = scene

        # extract radar data meta info
        self.rad_data_meta_info, \
        self.rad_timestamps_meta_info, \
        self.rad_calib_meta_info \
            = scene_rad_meta_info(config.root_dir_rad, scene)

        # extract camera data meta info
        self.cam_intrinsic_data_meta_info, \
        self.cam_extrinsic_data_meta_info, \
        self.cam_frame_names_meta_info, \
        self.cam_timestamps_meta_info \
            = scene_cam_meta_info(config.root_dir_cam, scene)

        # extract timestamp and radar mount info
        self.rad_timestamps = extract_timestamp_all_sensors(const.radar_location_attr, self.rad_timestamps_meta_info)
        self.cam_timestamps = extract_timestamp_all_sensors(const.camera_location_attr, self.cam_timestamps_meta_info)
        self.mount_param = extract_mounting_param_all_radars(const.radar_location_attr, self.rad_calib_meta_info)

        # extract radar and camera calibration matrix ( for projection on camera )
        self.extrinsic_calib_matrix_cam, \
        self.intrinsic_calib_matrix_cam \
            = extract_camera_calib_info(const.camera_location_attr, self.cam_extrinsic_data_meta_info, self.cam_intrinsic_data_meta_info)
        self.extrinsic_calib_matrix_rad = extract_SE3_extrinsic_matrix_all_radars(const.radar_location_attr, self.rad_calib_meta_info)

        # extract frame list
        self.num_frames = [ len(self.rad_timestamps[rad]) for rad in const.radar_location_attr ]   # number of frames radars
        self.radar_frames = create_frames_list_all_radars(const.radar_location_attr, self.num_frames, self.rad_data_meta_info)
        self.camera_frames = create_frame_list_all_cameras(const.camera_location_attr, self.cam_frame_names_meta_info)

        # create sync summary
        self.frame_summary, \
        self.imu_summary = sync_imu_all_radars(scene)
        _, self.cam_summary = sync_cam_all_radars(scene)

        # type of measurements needs to be selected
        self.meas_selection_function = meas_selection_function

        # maximum number of clusters
        self.nc = buffer_size * config.max_num_meas * ( config.num_samples + 1 )



class Config_viz:

    def __init__(self, radar, camera, conf, grid_prop):

        self.radar = radar
        self.camera = camera
        self.rad_id = const.radar_id[self.radar]
        self.cam_id = const.camera_id[self.camera]

        self.T_proj_rf_cf = create_projection_transformation(
            conf.extrinsic_calib_matrix_rad[const.radar_id[self.radar]], \
            conf.extrinsic_calib_matrix_cam[const.camera_id[self.camera]], \
            conf.intrinsic_calib_matrix_cam[const.camera_id[self.camera]])

        self.T_proj_vf_cf = create_projection_transformation(
            np.eye(4, dtype=np.float32), \
            conf.extrinsic_calib_matrix_cam[const.camera_id[self.camera]], \
            conf.intrinsic_calib_matrix_cam[const.camera_id[self.camera]])

        fov_coverage = FOV_grid_coverage(grid_prop, conf.scene)
        self.x_coord, self.y_coord = grid_prop.compute_xy_coordinates_from_scalar_ids(np.arange(grid_prop.num_cells, dtype=np.int32))
        self.xy_grid_coord = np.stack([self.x_coord, self.y_coord], axis=-1)
        self.fov_coverage_flag = fov_coverage.fov_coverage_flag



