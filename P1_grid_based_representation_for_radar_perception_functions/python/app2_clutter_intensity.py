import config, lib_const
import numpy as np
import matplotlib.pyplot as plt

from lib_read_data import (
    extract_radar_data,
    create_rad_frame_summary,
    extract_radar_mount_info, 
    select_all_invalid_meas_radar,
    scene_meta_info
)
from lib_grid import grid_properties, convert_grid_cart_to_polar

# ====================================================================================================================

dx = 0.1
dy = 0.1
grid_prop_cartesian = grid_properties( min_x = -10.0, max_x = 270.0, min_y = -110.0, max_y = 110.0, dx = dx, dy = dy)
grid_prop_polar = grid_properties( min_x = 0.0, max_x = 270.0, min_y = -90.0, max_y = 90.0, dx = dx, dy = dy)

# ====================================================================================================================

def compute_clutter_density(scene_names, grid_prop, mode='cartesian'):

    clutters = []
    prob_of_false_alarm = []
    num_frames = 0
    cellids_histogram = np.zeros((grid_prop.num_cells, ), dtype=np.float32) + 1e-10

    for scene_name in scene_names:
        print('scene name: ', scene_name)
        rad_data_meta_info, _, _ = scene_meta_info(config.root_dir, scene_name)
        #frame_summary, imu_summary = sync_imu_all_radars(scene_name)
        frame_summary = create_rad_frame_summary(scene_name)
        mount_param = extract_radar_mount_info(scene_name)

        for t in range(frame_summary.shape[0]):
            
            rad_meas, phd0, \
                = extract_radar_data(
                    t, rad_data_meta_info, select_all_invalid_meas_radar, 
                    frame_summary)
            rad_meas = rad_meas[:, :2]

            if mode == 'polar':
                clutters_polar_range, clutters_polar_theta = convert_grid_cart_to_polar(rad_meas[:, 0], rad_meas[:, 1])
                rad_meas = np.stack([clutters_polar_range, clutters_polar_theta], axis=-1)

            clutters.append(rad_meas)
            prob_of_false_alarm.append(phd0)

            # compute the cell ids
            cellids_meas = grid_prop.compute_scalar_ids_from_xy_coordinates(rad_meas[:, 0], rad_meas[:, 1])
            cellids_histogram[cellids_meas] = cellids_histogram[cellids_meas] + np.array(lib_const.false_alarm_probability_vals)[phd0.astype(int)] * 0.01
            # cellids_histogram[cellids_meas] = cellids_histogram[cellids_meas] + 1.0
        
        num_frames += frame_summary.shape[0]
    clutters = np.concatenate(clutters, axis=0) 
    prob_of_false_alarm = np.concatenate(prob_of_false_alarm, axis=0) 

    cellids = np.arange(grid_prop.num_cells, dtype=np.int32)
    cellxids, cellyids = grid_prop.compute_xy_ids_from_scalar_ids(cellids)
    cellxyimg = np.zeros((grid_prop.num_cells_x, grid_prop.num_cells_y), dtype=np.float32)
    cellxyimg[cellxids, cellyids] = cellids_histogram

    return (
        clutters,
        prob_of_false_alarm,
        num_frames,
        cellids_histogram,
        cellxyimg
    )

# ====================================================================================================================

# compute windowed average
def compute_windowed_sum(win_rows, win_cols, cellxyimg):
    offset_rows = win_rows // 2
    offset_cols = win_cols // 2

    num_rows = cellxyimg.shape[0] + 2*offset_rows
    num_cols = cellxyimg.shape[1] + 2*offset_cols

    conv2d_input_img = np.zeros((num_rows, num_cols), dtype=np.float32)
    conv2d_output_img = np.zeros((num_rows, num_cols), dtype=np.float32)
    conv2d_input_img[offset_rows:num_rows-offset_rows, offset_cols:num_cols-offset_cols] = cellxyimg

    
    for i in range(offset_rows, offset_rows + cellxyimg.shape[0]):
        startx_idx = i - offset_rows
        endx_idx = i + offset_rows + 1

        for j in range(offset_cols, offset_cols + cellxyimg.shape[1]):
            starty_idx = j - offset_cols
            endy_idx = j + offset_cols + 1

            window = conv2d_input_img[startx_idx:endx_idx, starty_idx:endy_idx]
            val = np.sum(window)
            conv2d_output_img[i, j] = val

    return conv2d_output_img[offset_rows:num_rows-offset_rows, offset_cols:num_cols-offset_cols]

# ====================================================================================================================

clutters_cart, _, num_frames, cellids_histogram_cart, cellxyimg_cart = compute_clutter_density(config.scene_names, grid_prop_cartesian, mode='cartesian')
clutters_polar, _, num_frames, cellids_histogram_polar, cellxyimg_polar = compute_clutter_density(config.scene_names, grid_prop_polar, mode='polar')

# ====================================================================================================================

win_rows = 51
win_cols = 51
conv2d_output_img_cart = compute_windowed_sum(win_rows, win_cols, cellxyimg_cart)
area = win_rows * dx * win_cols * dy
conv2d_output_img_cart = conv2d_output_img_cart / (area * num_frames)
print(conv2d_output_img_cart.shape)

# write as csv file
#np.savetxt('output.csv', conv2d_output_img_cart, delimiter=',')

# ====================================================================================================================

def set_plot_properties(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return ax

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.scatter(clutters_cart[:, 0], clutters_cart[:, 1], s=0.1, color='red', marker='.', label='clutter pxpy measurements')

fig, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.set_xlabel('range (m)')
ax1.set_ylabel('theta (deg)')
ax1.scatter(clutters_polar[:, 0], clutters_polar[:, 1], s=0.1, color='red', marker='.', label='clutter range theta measurements')

fig, ax3 = plt.subplots()
ax3.set_aspect('equal')
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.imshow(conv2d_output_img_cart)

plt.show()