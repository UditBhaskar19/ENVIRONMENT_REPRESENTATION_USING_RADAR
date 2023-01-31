# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Config
# =====================================================================================================================

root_dir = '../../Datasets/radarscenes'

# ransac parameters
ransac_min_num_samples = 2
ransac_error_margin = 0.25
ransac_num_iterations = 30
inlier_ratio_threshold = 0.6

# parameters for gating stationary measurements
gamma_stationary = 1.5
gamma_sq_vx_vy = 2**2


# sampling properties
num_particles = 200
sampling_variance = 0.2

# upper bound on the number of tracked cells
max_num_cells = 10000

process_noise = 0.05
max_noise = 5.0


