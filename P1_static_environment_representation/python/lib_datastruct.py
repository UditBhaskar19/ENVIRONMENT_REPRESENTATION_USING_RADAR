# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Data structures
# =====================================================================================================================

import numpy as np

class cell_states:
    """ cell states class """
    def __init__(self, max_num_cells, num_radars):

        self.num_tracked_cells = np.zeros((num_radars, ), dtype=np.uint32)
        self.cellids = np.zeros((num_radars, max_num_cells), dtype=np.uint32)
        self.logodds = np.zeros((num_radars, max_num_cells), dtype=np.float32)
        self.mu_px = np.zeros((num_radars, max_num_cells), dtype=np.float32)
        self.mu_py = np.zeros((num_radars, max_num_cells), dtype=np.float32)
        self.var_px = np.zeros((num_radars, max_num_cells), dtype=np.float32)
        self.var_py = np.zeros((num_radars, max_num_cells), dtype=np.float32)

        self.num_tracked_cells_fus = 0
        self.cellids_fus = np.zeros((num_radars*max_num_cells, ), dtype=np.uint32)
        self.logodds_fus = np.zeros((num_radars*max_num_cells, ), dtype=np.float32)
        self.mu_px_fus = np.zeros((num_radars*max_num_cells, ), dtype=np.float32)
        self.mu_py_fus = np.zeros((num_radars*max_num_cells, ), dtype=np.float32)

    # -----------------------------------------------------------------------

    def get_logodds_sensor_i(self, radarid):
        return self.logodds[radarid, :self.num_tracked_cells[radarid]]

    def get_cellids_sensor_i(self, radarid):
        return self.cellids[radarid, :self.num_tracked_cells[radarid]]

    def get_px_sensor_i(self, radarid):
        return self.mu_px[radarid, :self.num_tracked_cells[radarid]]

    def get_py_sensor_i(self, radarid):
        return self.mu_py[radarid, :self.num_tracked_cells[radarid]]

    def get_var_px_sensor_i(self, radarid):
        return self.var_px[radarid, :self.num_tracked_cells[radarid]]

    def get_var_py_sensor_i(self, radarid):
        return self.var_py[radarid, :self.num_tracked_cells[radarid]]

    # -----------------------------------------------------------------------

    def get_fused_logodds(self):
        return self.logodds_fus[:self.num_tracked_cells_fus]

    def get_fused_cellids(self):
        return self.cellids_fus[:self.num_tracked_cells_fus]

    def get_fused_px(self):
        return self.mu_px_fus[:self.num_tracked_cells_fus]

    def get_fused_py(self):
        return self.mu_py_fus[:self.num_tracked_cells_fus]

    # -----------------------------------------------------------------------

    def set_num_tracked_cells_sensor_i(self, radarid, n):
        self.num_tracked_cells[radarid] = n

    def set_logodds_sensor_i(self, radarid, logodds):
        self.logodds[radarid, :self.num_tracked_cells[radarid]] = logodds

    def set_cellids_sensor_i(self, radarid, cellids):
        self.cellids[radarid, :self.num_tracked_cells[radarid]] = cellids

    def set_px_sensor_i(self, radarid, xcoord):
        self.mu_px[radarid, :self.num_tracked_cells[radarid]] = xcoord

    def set_py_sensor_i(self, radarid, ycoord):
        self.mu_py[radarid, :self.num_tracked_cells[radarid]] = ycoord

    def set_var_px_sensor_i(self, radarid, var_xcoord):
        self.var_px[radarid, :self.num_tracked_cells[radarid]] = var_xcoord

    def set_var_py_sensor_i(self, radarid, var_ycoord):
        self.var_py[radarid, :self.num_tracked_cells[radarid]] = var_ycoord

    # -----------------------------------------------------------------------

    def set_num_fused_tracked_cells(self, n):
        self.num_tracked_cells_fus = n

    def set_fused_logodds(self, logodds):
        self.logodds_fus[:self.num_tracked_cells_fus] = logodds

    def set_fused_cellids(self, cellids):
        self.cellids_fus[:self.num_tracked_cells_fus] = cellids

    def set_fused_px(self, cellxcoord):
        self.mu_px_fus[:self.num_tracked_cells_fus] = cellxcoord

    def set_fused_py(self, cellycoord):
        self.mu_py_fus[:self.num_tracked_cells_fus] = cellycoord

# ====================================================================================================================

class meas_hist_buffer:
    """ A naive implementation of measurement hist buffer in case 
    we would like to store a sequence of radar frames and pose history """
    def __init__(self, buff_size):
        self.top = -1
        self.buff_size = buff_size

        self.meas_xcoord = np.array([])
        self.meas_ycoord = np.array([])

        self.num_meas = np.zeros((buff_size, ), dtype=np.uint32)
        self.radar_id = np.zeros((buff_size, ), dtype=np.uint32)
        self.timestamp = np.zeros((buff_size, ), dtype=np.float32)

        self.pose_x = np.zeros((buff_size, ), dtype=np.float32)
        self.pose_y = np.zeros((buff_size, ), dtype=np.float32)
        self.pose_yaw = np.zeros((buff_size, ), dtype=np.float32)
        
    def is_empty(self):
        return self.top == -1
    
    def is_full(self):
        return self.top == ( self.buff_size - 1 )
    
    def update_buffer(
        self, 
        meas_xcoord, meas_ycoord, 
        num_meas, radar_id, timestamp,
        pose_x, pose_y, pose_yaw):

        if self.is_full():
            self.meas_xcoord = np.concatenate([self.meas_xcoord[self.num_meas[0]:], meas_xcoord], axis=0)
            self.meas_ycoord = np.concatenate([self.meas_ycoord[self.num_meas[0]:], meas_ycoord], axis=0)

            self.num_meas[:self.buff_size-1] = self.num_meas[1:self.buff_size]
            self.radar_id[:self.buff_size-1] = self.radar_id[1:self.buff_size]
            self.timestamp[:self.buff_size-1] = self.timestamp[1:self.buff_size]

            self.pose_x[:self.buff_size-1] = self.pose_x[1:self.buff_size]
            self.pose_y[:self.buff_size-1] = self.pose_y[1:self.buff_size]
            self.pose_yaw[:self.buff_size-1] = self.pose_yaw[1:self.buff_size]
        else:
            self.meas_xcoord = np.concatenate([self.meas_xcoord, meas_xcoord], axis=0)
            self.meas_ycoord = np.concatenate([self.meas_ycoord, meas_ycoord], axis=0)
            self.top += 1

        self.radar_id[self.top] = radar_id
        self.timestamp[self.top] = timestamp
        self.num_meas[self.top] = num_meas

        self.pose_x[self.top] = pose_x
        self.pose_y[self.top] = pose_y
        self.pose_yaw[self.top] = pose_yaw


        