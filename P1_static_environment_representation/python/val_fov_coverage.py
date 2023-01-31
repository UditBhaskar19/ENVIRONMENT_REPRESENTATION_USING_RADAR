# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validation script lib_fov_coverage.py
# =====================================================================================================================

import lib_const
import numpy as np
import matplotlib.pyplot as plt
from lib_grid import grid_properties, FOV_grid_coverage

def compute_single_radar_zone(radarid, fov_coverage_flag):
    cond = np.ones(fov_coverage_flag.shape[-1], dtype=np.bool8)
    for i in radarid:
        cond = np.logical_and(cond, fov_coverage_flag[i])
    for i in range(lib_const.num_radars):
        if i not in radarid:
            cond = np.logical_and(cond, np.logical_not(fov_coverage_flag[i]))
    return cond


def plot_ego_motion(
    grid_points, FOV_coverage_points,
    idx, color, radar_id, ax):
    i, j = idx
    ax[i,j].scatter(grid_points[:,0], grid_points[:,1], s=1, color='yellow', marker='.', label='grid points')
    ax[i,j].scatter(FOV_coverage_points[:,0], FOV_coverage_points[:,1], s=2, color=color, marker='.', label='radar ' + str(radar_id) + ' fov')
    ax[i,j].set_aspect('equal')
    ax[i,j].legend(loc='upper right')
    ax[i,j].set_xlabel('x (m)')
    ax[i,j].set_ylabel('y (m)')
    return ax


if __name__ == '__main__':

    print('Validating grid.py !!!!!! ........')

    GridCartesian = grid_properties(
        min_x = -90.0, 
        max_x = 120.0, 
        min_y = -120.0, 
        max_y = 120.0, 
        dx = 1, 
        dy = 1
    )

    fov_coverage = FOV_grid_coverage(GridCartesian)
    xy_grid_coord = np.stack([fov_coverage.x_coord, fov_coverage.y_coord], axis=-1)


    rad1_coverage_xycoord = xy_grid_coord[fov_coverage.fov_coverage_flag[0]]    # identify coverage by radar 1
    rad2_coverage_xycoord = xy_grid_coord[fov_coverage.fov_coverage_flag[1]]    # identify coverage by radar 2
    rad3_coverage_xycoord = xy_grid_coord[fov_coverage.fov_coverage_flag[2]]    # identify coverage by radar 3
    rad4_coverage_xycoord = xy_grid_coord[fov_coverage.fov_coverage_flag[3]]    # identify coverage by radar 4

    rad1_only_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([0], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 1 only
    rad4_only_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([3], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 4 only
    rad12_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([0,1], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 1 and 2
    rad23_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([1,2], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 2 and 3
    rad34_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([2,3], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 3 and 4
    rad123_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([0,1,2], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 1, 2 and 3
    rad234_coverage_xycoord = xy_grid_coord[compute_single_radar_zone([1,2,3], fov_coverage.fov_coverage_flag)]  # identify coverage by radar 2, 3 and 4



    # individual fov plots
    fig, ax = plt.subplots(2,2)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    ax = plot_ego_motion(xy_grid_coord, rad1_coverage_xycoord, (0,0), 'black', 1, ax)
    ax = plot_ego_motion(xy_grid_coord, rad2_coverage_xycoord, (1,0), 'green', 2, ax)
    ax = plot_ego_motion(xy_grid_coord, rad3_coverage_xycoord, (0,1), 'red',   3, ax)
    ax = plot_ego_motion(xy_grid_coord, rad4_coverage_xycoord, (1,1), 'blue',  4, ax)
    fig.suptitle('Individual Radar FOV', fontsize=16)


    # fov coverage plot all radars
    fig, ax = plt.subplots(1,1)
    markersize = 5
    ax.scatter(xy_grid_coord[:,0], xy_grid_coord[:,1], s=1, color='yellow', marker='.', label='grid points')
    ax.scatter(rad1_coverage_xycoord[:,0], rad1_coverage_xycoord[:,1], s=markersize, color='red', marker='.', label='radar 1 fov')
    ax.scatter(rad2_coverage_xycoord[:,0], rad2_coverage_xycoord[:,1], s=markersize, color='firebrick', marker='.', label='radar 2 fov')
    ax.scatter(rad3_coverage_xycoord[:,0], rad3_coverage_xycoord[:,1], s=markersize, color='maroon', marker='.', label='radar 3 fov')
    ax.scatter(rad4_coverage_xycoord[:,0], rad4_coverage_xycoord[:,1], s=markersize, color='darkred', marker='.', label='radar 4 fov')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.title('All Radar FOV', fontsize=16)


    # fov coverage plot all radars
    fig, ax = plt.subplots(1,1)
    markersize = 6
    ax.scatter(xy_grid_coord[:,0], xy_grid_coord[:,1], s=1, color='white', marker='.', label='grid points')
    ax.scatter(rad1_only_coverage_xycoord[:,0], rad1_only_coverage_xycoord[:,1], s=markersize, color='blue', marker='.', label='radar 1 only fov')
    ax.scatter(rad4_only_coverage_xycoord[:,0], rad4_only_coverage_xycoord[:,1], s=markersize, color='red', marker='.', label='radar 4 only fov')
    ax.scatter(rad12_coverage_xycoord[:,0], rad12_coverage_xycoord[:,1], s=markersize, color='green', marker='.', label='radar 1 and 2 only fov')
    ax.scatter(rad23_coverage_xycoord[:,0], rad23_coverage_xycoord[:,1], s=markersize, color='magenta', marker='.', label='radar 2 and 3 only fov')
    ax.scatter(rad34_coverage_xycoord[:,0], rad34_coverage_xycoord[:,1], s=markersize, color='orange', marker='.', label='radar 3 and 4 only fov')
    ax.scatter(rad123_coverage_xycoord[:,0], rad123_coverage_xycoord[:,1], s=markersize, color='black', marker='.', label='radar 1, 2 and 3 only fov')
    ax.scatter(rad234_coverage_xycoord[:,0], rad234_coverage_xycoord[:,1], s=markersize, color='darkcyan', marker='.', label='radar 2, 3 and 4 only fov')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.title('FOV Zones', fontsize=16)

    plt.show()


    