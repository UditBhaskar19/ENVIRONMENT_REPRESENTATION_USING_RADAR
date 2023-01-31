# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validation script lib_log_odds.py
# =====================================================================================================================

import os, lib_const, config
import numpy as np
import matplotlib.pyplot as plt
from lib_read_data import (
    parse_csv_file, 
    select_radar_mount_parameters
)
from lib_grid import (
    grid_properties, 
    convert_meas_polar_to_cartesian
)
from lib_meas_selection import (
    coordinate_transform_px_py,
    select_stationary_measurements
)
from lib_log_odds import (
    compute_meas_log_likelihood,
    inflate_prob_and_compute_log_odds,
    compute_prob_from_log_odds
)


def plot_meas(meas_x, meas_y):
    _, ax = plt.subplots(1,1)
    ax.scatter(meas_y, meas_x, s=0.1, color='red', marker='.', label='cell coordinates')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('x (m)')
    plt.title('radar measurements', fontsize=16)


def plot_res(res_map_x):
    x = np.linspace(start=-10, stop=120, num=1000)
    dx1 = res_map_x.compute_resolution_exponential(x)
    dx2 = res_map_x.compute_resolution_linear(x)
    _, ax = plt.subplots(1,1)
    ax.plot(x, dx1, color='red', label='resolution exp')
    ax.plot(x, dx2, color='green', label='resolution linear')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('dx (m)')
    plt.title('variation in resolution', fontsize=16)


if __name__ == '__main__':

    print('Validating log_odds.py !!!!!! ........')

    GridCart = grid_properties( min_x = 0.0, max_x = 110.0, min_y = -50, max_y = 50, dx = 0.2, dy = 0.2 )
    
    

    scene = '105'       # 105, 108 
    sensor = 'radar3'
    frameid = 1010

    data_dir = os.path.join(config.root_dir, scene, sensor)
    file_timestamp_and_odom = os.path.join(data_dir, 'timestamp_and_odom.csv')
    _, data_timestamp_and_odom = parse_csv_file(file_timestamp_and_odom)                 # extract odometry data and timestamps
    radar_mount_param = select_radar_mount_parameters(sensor)                            # extract radar mount parameters
    
    # extract the radar frame data
    file_meas = os.path.join(data_dir, str(frameid + 1) + '.csv')
    _, rad_meas = parse_csv_file(file_meas)

    # extract time stamps and odom
    vx_odom = data_timestamp_and_odom[frameid, lib_const.odom_attr['vx_odom']]
    yawrate_odom = data_timestamp_and_odom[frameid, lib_const.odom_attr['yawrate_odom']]

    # select stationary measurements
    meas_stationary = select_stationary_measurements(
        rad_meas, radar_mount_param,
        vx_odom, yawrate_odom)

    # convert meas from polar to cartesian
    meas_stationary_x, \
    meas_stationary_y \
        = convert_meas_polar_to_cartesian(
            meas_stationary[:, lib_const.rad_meas_attr['range']], 
            meas_stationary[:, lib_const.rad_meas_attr['azimuth']])

    # cts from sensor frame to vehicle frame
    tx, ty, theta = radar_mount_param
    meas_stationary_x, \
    meas_stationary_y \
        = coordinate_transform_px_py(
            meas_stationary_x, 
            meas_stationary_y, 
            tx, ty, theta)

    # compute measuremeny log-odds
    prob, cellids, xcoord, ycoord = compute_meas_log_likelihood(
        meas_stationary_x,
        meas_stationary_y, 
        0.5,
        1000,
        GridCart)

    # compute log odds
    log_odds = inflate_prob_and_compute_log_odds(prob)

    # for visualization
    cellxid, cellyid = GridCart.compute_xy_ids_from_scalar_ids(cellids)
    # prob_pred_img = 0.5 + np.zeros((GridCart.num_cells_x, GridCart.num_cells_y), dtype=np.float32)
    # prob_pred_img[cellxid, cellyid] = compute_prob_from_log_odds(log_odds)
    prob_pred_img = np.zeros((GridCart.num_cells_x, GridCart.num_cells_y), dtype=np.float32)
    prob_pred_img[cellxid, cellyid] = 2 * compute_prob_from_log_odds(log_odds) - 1
    

    plot_meas(xcoord, ycoord)
    
    matfig = plt.figure(figsize=(15,10))
    img = prob_pred_img
    img = np.flip(img, axis=0)
    plt.matshow(img, fignum=matfig.number)
    plt.show()


