# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : perform clustering for front radar and rear-left radar
# =====================================================================================================================

import os, const, config
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
    accumulate_radar_frames,
    perform_clustering
)
from lib_datastruct import meas_hist_buffer
from lib_clustering import DBSCAN, compute_cov_ellipse 
from lib_grid import grid_properties, FOV_grid_coverage
from lib_functions import create_projection_transformation, project_radar_on_camera

# ====================================================================================================================

# supported scenes : 0655, 0103, 0796, 1077, 1094
# supported cameras : front, rear

scene = 'scene-1077'
radar_front = 'front'
camera_front = 'front'
radar_rear = 'rear_left'
camera_rear = 'rear'

buffer_size = 10
eps_dist = 4.5
eps_num_pts = 4

grid_prop = grid_properties( min_x = -200.0, max_x = 200.0, min_y = -200.0, max_y = 200.0, dx = 1.0, dy = 1.0)
conf = Config(scene, select_moving_all_meas_radar, buffer_size)
conf_viz_front = Config_viz(radar_front, camera_front, conf, grid_prop)
conf_viz_rear = Config_viz(radar_rear, camera_rear, conf, grid_prop)

meas_hist = [ meas_hist_buffer(buffer_size, config.max_num_meas, config.num_samples) for i in range(const.num_radars) ]
meas_clusters_front = DBSCAN(conf.nc, eps_dist, eps_num_pts)
meas_clusters_rear = DBSCAN(conf.nc, eps_dist, eps_num_pts)

timestamp_rad_prev = 0
trigger = np.ones((const.num_radars, ), dtype=np.bool8)

# ====================================================================================================================

def set_plot_properties(ax):
    ax[0,0].set_xlim([-160, 0])
    ax[0,0].set_ylim([-30, 30])
    ax[0,0].set_aspect('equal')

    ax[0,1].set_xlim([0, 160])
    ax[0,1].set_ylim([-30, 30])
    ax[0,1].set_aspect('equal')

    ax[1,0].set_xlim([0, 1600])
    ax[1,0].set_ylim([0, 900])
    ax[1,0].set_aspect('equal')

    ax[1,1].set_xlim([0, 1600])
    ax[1,1].set_ylim([0, 900])
    ax[1,1].set_aspect('equal')
    return ax

# ====================================================================================================================

def project_rad_clusters_on_cam(meas_clusters, conf, conf_viz, t):

    # closest front camera frame id
    closest_front_cam_info = conf.cam_summary[conf_viz.cam_id][t]
    cam_frame = conf.camera_frames[conf_viz.camera][int(closest_front_cam_info[1])]

    # project the radar point on the camera frame
    clstr_px = meas_clusters.meas_vector[:meas_clusters.num_clusters, 0]
    clstr_py = meas_clusters.meas_vector[:meas_clusters.num_clusters, 1]
    clstr_vx = meas_clusters.meas_vector[:meas_clusters.num_clusters, 2]
    clstr_vy = meas_clusters.meas_vector[:meas_clusters.num_clusters, 3]
    clstr_pxpyvxvy = np.stack([clstr_px, clstr_py, clstr_vx, clstr_vy], axis=-1)
    rad_meas_proj = project_radar_on_camera(conf_viz.T_proj_vf_cf, clstr_pxpyvxvy[:, :2])

    # select points within the image
    condition1 = np.logical_and(rad_meas_proj[:, 0] >= 0, rad_meas_proj[:, 0] < 1600)
    condition2 = np.logical_and(rad_meas_proj[:, 1] >= 0, rad_meas_proj[:, 1] < 900)
    condition = np.logical_and(condition1, condition2)
    rad_meas_proj = rad_meas_proj[condition]
    return rad_meas_proj, cam_frame, clstr_pxpyvxvy


def compute_cluster_boundary(meas_clusters):
    points_boundary = []
    for i in range(meas_clusters.num_clusters):
        cluster_center_x, cluster_center_y = meas_clusters.meas_vector[i, 0], meas_clusters.meas_vector[i, 1]
        points, _ = compute_cov_ellipse( np.stack([cluster_center_x, cluster_center_y], axis=-1), meas_clusters.shape_covariance[i, :2, :2], 2.2, 100)
        points_boundary.append(points)
    if meas_clusters.num_clusters > 0:
        points_boundary = np.concatenate(points_boundary, axis=0)
    return points_boundary

# ====================================================================================================================

class animate(object):

    def __init__(self, ax):

        self.meas_hist = meas_hist
        self.trigger = trigger
        self.meas_clusters_front = meas_clusters_front
        self.meas_clusters_rear = meas_clusters_rear
        self.timestamp_rad_prev = timestamp_rad_prev

        self.ax = ax
        self.scale = 200

        self.cluster_center_xy_rear = ax[0,0].scatter([],[], s=50, color='blue', marker='x', label='cluster center')
        self.cluster_boundary_rear = ax[0,0].scatter([],[], s=1, color='black', marker='.')
        self.cluster_vr_rear = ax[0,0].quiver([], [], [], [])
        self.radar_fov_rear = self.ax[0,0].scatter([],[], s=1, color='red', marker='.', alpha=0.2)
        self.radar_fov_rear.set_offsets(conf_viz_rear.xy_grid_coord[conf_viz_rear.fov_coverage_flag[2]])
        self.meas_xy_rear = self.ax[0,0].scatter([],[], s=10, color='green', marker='x', label='measurements')

        self.cluster_center_xy_front = ax[0,1].scatter([],[], s=50, color='blue', marker='x', label='cluster center')
        self.cluster_boundary_front = ax[0,1].scatter([],[], s=1, color='black', marker='.')
        self.cluster_vr_front = ax[0,1].quiver([], [], [], [])
        self.radar_fov_front = self.ax[0,1].scatter([],[], s=1, color='blue', marker='.', alpha=0.2)
        self.radar_fov_front.set_offsets(conf_viz_front.xy_grid_coord[conf_viz_front.fov_coverage_flag[0]])
        self.meas_xy_front = self.ax[0,1].scatter([],[], s=10, color='red', marker='x', label='measurements')

        self.meas_proj_xy_rear = self.ax[1,0].scatter([],[], s=50, color='yellow', marker='x')
        self.meas_proj_xy_front = self.ax[1,1].scatter([],[], s=50, color='red', marker='x')

    # --------------------------------------------------------------------------------------------------------------------------------------------------

    def __call__(self, t):

        # accumulate measurement in buffer
        self.meas_hist, self.trigger, self.timestamp_rad_prev, radar_id = accumulate_radar_frames(t, conf, self.meas_hist, self.trigger, self.timestamp_rad_prev)

        # perform clustering of dynamic measurements
        self.meas_clusters_front, meas_vector_front = perform_clustering(radar_front, self.trigger, self.meas_hist, self.meas_clusters_front)
        self.meas_clusters_rear, meas_vector_rear = perform_clustering(radar_rear, self.trigger, self.meas_hist, self.meas_clusters_rear)

        print('frame_idx: ', t, ' time: ', self.timestamp_rad_prev)

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        if radar_id == 0:

            rad_meas_proj, cam_frame, clstr_pxpyvxvy = project_rad_clusters_on_cam(self.meas_clusters_front, conf, conf_viz_front, t)
            points_boundary = compute_cluster_boundary(self.meas_clusters_front)
            
            self.meas_xy_front.set_offsets(meas_vector_front[:, :2])
            self.cluster_vr_front.remove()
            self.cluster_vr_front = ax[0,1].quiver( clstr_pxpyvxvy[:, 0], clstr_pxpyvxvy[:, 1], clstr_pxpyvxvy[:, 2], clstr_pxpyvxvy[:, 3], scale=self.scale, color='k', width=0.004)
            self.cluster_center_xy_front.set_offsets(clstr_pxpyvxvy[:, :2])
            if len(points_boundary) == 0:
                self.cluster_boundary_front.remove()
                self.cluster_boundary_front = ax[0,1].scatter([],[], s=1, color='black', marker='.')
            else:
                self.cluster_boundary_front.set_offsets(points_boundary[:, :2])
            self.ax[0,1].set_title(str(conf.scene) + ': Front Radar ' + ': time ' + f'{self.timestamp_rad_prev:.2f}' + ' sec', fontsize=12)

            # visualize cluster on image
            img = Image.open(cam_frame)
            self.ax[1,1].clear()
            self.ax[1,1].imshow(img)

            if rad_meas_proj.shape[0] > 0:
                self.ax[1,1].scatter(rad_meas_proj[:,0], rad_meas_proj[:,1], s=20, color='red', marker='x', label='radar measurements')

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        if radar_id == 2:

            rad_meas_proj, cam_frame, clstr_pxpyvxvy = project_rad_clusters_on_cam(self.meas_clusters_rear, conf, conf_viz_rear, t)
            points_boundary = compute_cluster_boundary(self.meas_clusters_rear)

            self.meas_xy_rear.set_offsets(meas_vector_rear[:, :2])
            self.cluster_vr_rear.remove()
            self.cluster_vr_rear = ax[0,0].quiver( clstr_pxpyvxvy[:, 0], clstr_pxpyvxvy[:, 1], clstr_pxpyvxvy[:, 2], clstr_pxpyvxvy[:, 3], scale=self.scale, color='k', width=0.004)
            self.cluster_center_xy_rear.set_offsets(clstr_pxpyvxvy[:, :2])
            if len(points_boundary) == 0:
                self.cluster_boundary_rear.remove()
                self.cluster_boundary_rear = ax[0,0].scatter([],[], s=1, color='black', marker='.')
            else:
                self.cluster_boundary_rear.set_offsets(points_boundary[:, :2])
            self.ax[0,0].set_title(str(conf.scene) + ': Rear Radar ' + str(radar_id) + ': time ' + f'{self.timestamp_rad_prev:.2f}' + ' sec', fontsize=12)

            # visualize cluster on image
            img = Image.open(cam_frame)
            self.ax[1,0].clear()
            self.ax[1,0].imshow(img)

            if rad_meas_proj.shape[0] > 0:
                self.ax[1,0].scatter(rad_meas_proj[:,0], rad_meas_proj[:,1], s=20, color='yellow', marker='x', label='radar measurements')

# ===================================================================================================================

fig, ax = plt.subplots(nrows=2, ncols=2)
ax = set_plot_properties(ax)
integrated_scans = animate(ax)


anim = FuncAnimation(fig, integrated_scans, frames=np.arange(conf.frame_summary.shape[0]), interval=0.002, repeat=False)
plt.show()


# anim = FuncAnimation(fig, integrated_scans, frames=np.arange(conf.frame_summary.shape[0] // 2 ), interval=1, repeat=False)
# plt.show()
# anim.save('result_videos/radar_clusters_1094_full.gif', writer='pillow', fps=50)