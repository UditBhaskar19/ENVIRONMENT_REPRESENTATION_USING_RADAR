# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Configurations
# =====================================================================================================================
import numpy as np

# sensor csv data directory 
root_dir_can = '../../Datasets/nuscenes_mini_radar'
root_dir_rad = '../../Datasets/nuscenes_mini_radar'
root_dir_cam = '../../Datasets/nuscenes_mini_camera'

max_num_meas = 200
meas_dim = 4

# measurement noise for computing gating boundary
R = np.eye(meas_dim, dtype=np.float32)
R[0,0] = 0.794
R[1,1] = 0.794
R[2,2] = 0.616
R[3,3] = 0.011

# number of extra samples generated from each radar measurement
# set it to 0 if extra sampling should not be used
num_samples = 0

# sampling covariance
sampling_covariance = np.eye(meas_dim, dtype=np.float32)
sampling_covariance[0,0] = 0.5
sampling_covariance[1,1] = 0.5
sampling_covariance[2,2] = 0.05
sampling_covariance[3,3] = 0.05

# clustering parameters
cluster_shape = np.array([[2,0],[0,2]], dtype=np.float32)