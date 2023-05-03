# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Visualize radar scans and project the scans on the image (front and rear-left radar) 
# =====================================================================================================================

import os, config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from config_object import Config, Config_viz

from lib_read_data import (
    select_moving_all_meas_radar, 
    select_moving_all_valid_meas_radar,
    select_all_meas_radar
)
from lib_main_functions import (
    extract_sensor_data,
    coordinate_transform
)
from lib_sampling import generate_radar_meas 
from lib_grid import grid_properties, FOV_grid_coverage
from lib_functions import create_projection_transformation, project_radar_on_camera

# ====================================================================================================================

scene = 'scene-1077'          #scene-0655, scene-0103, scene-0796, scene-1077, scene-1094
radar_front = 'front'
camera_front = 'front'
radar_rear = 'rear_left'
camera_rear = 'rear'

buffer_size = 10
grid_prop = grid_properties( min_x = -200.0, max_x = 200.0, min_y = -200.0, max_y = 200.0, dx = 1.0, dy = 1.0)
conf = Config(scene, select_moving_all_meas_radar, buffer_size)
conf_viz_front = Config_viz(radar_front, camera_front, conf, grid_prop)
conf_viz_rear = Config_viz(radar_rear, camera_rear, conf, grid_prop)

# ====================================================================================================================

def set_plot_properties(ax):
    ax[0,0].set_xlim([-180, 0])
    ax[0,0].set_ylim([-50, 50])
    ax[0,0].set_aspect('equal')

    ax[0,1].set_xlim([0, 180])
    ax[0,1].set_ylim([-50, 50])
    ax[0,1].set_aspect('equal')

    ax[1,0].set_xlim([0, 1600])
    ax[1,0].set_ylim([0, 900])
    ax[1,0].set_aspect('equal')

    ax[1,1].set_xlim([0, 1600])
    ax[1,1].set_ylim([0, 900])
    ax[1,1].set_aspect('equal')
    return ax

# ====================================================================================================================

def project_rad_on_cam(rad_meas, conf, conf_viz, t):

    # closest front camera frame id
    closest_front_cam_info = conf.cam_summary[conf_viz.cam_id][t]
    cam_frame = conf.camera_frames[conf_viz.camera][int(closest_front_cam_info[1])]

    # project the radar point on the camera frame
    rad_meas_proj = project_radar_on_camera(conf_viz.T_proj_vf_cf, rad_meas[:, :2])

    # select points within the image
    condition1 = np.logical_and(rad_meas_proj[:, 0] >= 0, rad_meas_proj[:, 0] < 1600)
    condition2 = np.logical_and(rad_meas_proj[:, 1] >= 0, rad_meas_proj[:, 1] < 900)
    condition = np.logical_and(condition1, condition2)
    rad_meas_proj = rad_meas_proj[condition]
    return rad_meas_proj, cam_frame

# ====================================================================================================================

class animate(object):

    def __init__(self, ax, conf, conf_viz_front, conf_viz_rear):

        self.ax = ax
        self.conf = conf
        self.conf_viz_front = conf_viz_front
        self.conf_viz_rear = conf_viz_rear
        
        self.scale = 100

        self.meas_vr_rear = ax[0,0].quiver([], [], [], [])
        self.radar_fov_rear = self.ax[0,0].scatter([],[], s=1, color='red', marker='.', alpha=0.2)
        self.radar_fov_rear.set_offsets(conf_viz_rear.xy_grid_coord[conf_viz_rear.fov_coverage_flag[2]])
        self.meas_xy_rear = self.ax[0,0].scatter([],[], s=50, color='green', marker='x', label='measurements')

        self.meas_vr_front = ax[0,1].quiver([], [], [], [])
        self.radar_fov_front = self.ax[0,1].scatter([],[], s=1, color='blue', marker='.', alpha=0.2)
        self.radar_fov_front.set_offsets(conf_viz_front.xy_grid_coord[conf_viz_front.fov_coverage_flag[0]])
        self.meas_xy_front = self.ax[0,1].scatter([],[], s=50, color='red', marker='x', label='measurements')

        self.meas_proj_xy_rear = self.ax[1,0].scatter([],[], s=50, color='yellow', marker='x')
        self.meas_proj_xy_front = self.ax[1,1].scatter([],[], s=50, color='red', marker='x')

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def __call__(self, t):

        # extract sensor data
        radar_id, radar_time_stamp, rad_meas_sf, rad_meas_rms, phd0, imu_meas = extract_sensor_data(t, self.conf)

        # extract the measurements
        rad_meas, meas_cov, are_samples = generate_radar_meas(rad_meas_sf, rad_meas_rms, config.sampling_covariance, config.num_samples)

        # coordinate transformation sf to vf
        rad_meas, meas_cov = coordinate_transform(
            rad_meas, meas_cov, self.conf.mount_param[radar_id, 0], 
            self.conf.mount_param[radar_id, 1], self.conf.mount_param[radar_id, 2])


        print('frame_idx: ', t, ' time: ', radar_time_stamp)

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        if radar_id == 0:

            rad_meas_proj, cam_frame = project_rad_on_cam(rad_meas, conf, conf_viz_front, t)
            
            self.meas_vr_front.remove()
            self.meas_vr_front = ax[0,1].quiver( rad_meas[:, 0], rad_meas[:, 1], rad_meas[:, 2], rad_meas[:, 3], scale=self.scale, color='k', width=0.004)
            self.meas_xy_front.set_offsets(rad_meas[:, :2])
            self.ax[0,1].set_title(str(self.conf.scene) + ': Front Radar ' + ': time ' + f'{radar_time_stamp:.2f}' + ' sec', fontsize=12)

            # visualize cluster on image
            img = Image.open(cam_frame)
            self.ax[1,1].clear()
            self.ax[1,1].imshow(img)

            if rad_meas_proj.shape[0] > 0:
                self.ax[1,1].scatter(rad_meas_proj[:,0], rad_meas_proj[:,1], s=20, color='red', marker='x', label='radar measurements')

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        if radar_id == 2:

            rad_meas_proj, cam_frame = project_rad_on_cam(rad_meas, conf, conf_viz_rear, t)
            
            self.meas_vr_rear.remove()
            self.meas_vr_rear = ax[0,0].quiver( rad_meas[:, 0], rad_meas[:, 1], rad_meas[:, 2], rad_meas[:, 3], scale=self.scale, color='k', width=0.004)
            self.meas_xy_rear.set_offsets(rad_meas[:, :2])
            self.ax[0,0].set_title(str(self.conf.scene) + ': Rear Radar ' + str(radar_id) + ': time ' + f'{radar_time_stamp:.2f}' + ' sec', fontsize=12)

            # visualize cluster on image
            img = Image.open(cam_frame)
            self.ax[1,0].clear()
            self.ax[1,0].imshow(img)

            if rad_meas_proj.shape[0] > 0:
                self.ax[1,0].scatter(rad_meas_proj[:,0], rad_meas_proj[:,1], s=20, color='yellow', marker='x', label='radar measurements')

# ===================================================================================================================

fig, ax = plt.subplots(nrows=2, ncols=2)
ax = set_plot_properties(ax)
integrated_scans = animate(ax, conf, conf_viz_front, conf_viz_rear)


anim = FuncAnimation(fig, integrated_scans, frames=np.arange(conf.frame_summary.shape[0]), interval=0.002, repeat=False)
plt.show()


# anim = FuncAnimation(fig, integrated_scans, frames=np.arange(conf.frame_summary.shape[0] // 10 ), interval=1, repeat=False)
# plt.show()
# anim.save('result_videos/radar_scans_1077_short.gif', writer='pillow', fps=50)