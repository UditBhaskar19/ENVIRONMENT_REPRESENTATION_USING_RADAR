# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Measurement generation

# function names
# - compute_measurement_covariances() : compute the measurement covariance matrix from meas variances
# - generate_samples() : Generate samples and sample covariances 
# - generate_radar_meas() : Generate more radar measurements
# =====================================================================================================================

import numpy as np
import config

# =====================================================================================================================

def compute_measurement_covariances(meas, var_px, var_py, var_vx, var_vy):
    """ compute the measurement covariance matrix from meas variances
    Inputs: var_px, var_py, var_vx, var_vy - each are an array of measurement variances of shape (m, ).
    Outputs: meas_cov - an array of measurement covariances of shape m x 4 x 4. Each entry corrosponds to (px, py, vx, vy)
    """
    R = np.zeros((meas.shape[0], config.meas_dim, config.meas_dim), dtype=np.float32)
    R[:,  0,  0] = var_px
    R[:,  1,  1] = var_py
    R[:,  2,  2] = var_vx
    R[:,  3,  3] = var_vy
    return R

# =====================================================================================================================

def generate_samples(
    meas,
    meas_cov, 
    sampling_cov, 
    num_samples):
    """ Generate samples and sample covariances 
    Inputs: meas - meas array of shape (m, 4). Each entry corrosponds to (px, py, vx, vy)
            meas_cov -  meas noise covariance of shape (m, 4, 4).
            sampling_cov - sampling covariance of shape (4, 4)
            num_samples - number of samples per measurement that has to be generated
    Outputs: samples - measurement samples of shape (n, 4). Each entry corrosponds to (px, py, vx, vy)
             samples_cov - covariance of the samples of shape (n, 4, 4).
    """
    samples = []
    samples_cov = []
    for i in range(meas.shape[0]):
        particles = np.random.multivariate_normal(meas[i], sampling_cov[i], num_samples)
        cov = np.repeat(np.expand_dims(meas_cov[i], axis=0), num_samples, axis=0)
        samples.append(particles)
        samples_cov.append( cov )
    samples = np.concatenate( samples, axis=0 )
    samples_cov = np.concatenate( samples_cov, axis=0 )
    return samples, samples_cov

# =====================================================================================================================

def generate_radar_meas(
    rad_meas, 
    rad_meas_rms, 
    sampling_cov,
    num_samples):
    """ Generate more radar measurements
    Inputs: rad_meas - an array of vectors of shape (n, 4). Each attributes corrosponds to (px, py, vx, vy)
          : rad_meas_rms - an array of vectors of shape (n, 4). Each attributes corrosponds to (px_rms, py_rms, vx_rms, vy_rms)
          : sampling_cov - sampling covariance of shape (4, 4)
          : num_samples - number of samples per measurement that has to be generated
    Outputs: rad_meas - radar measurements + generated samples : an array of vector of shape ( num_meas*(1 + num_samples), 4 )
           : meas_cov - noise covariances of shape ( num_meas*(1 + num_samples), 4, 4 )
           : are_samples - a boolean flag of shape ( num_meas*(1 + num_samples), ) indicating if a measurement is generated or an actual
    """    
    are_samples = np.zeros((rad_meas.shape[0], ), dtype=np.bool8)
    if rad_meas.shape[0] == 0: meas_cov = np.zeros((0, 4, 4), dtype=np.float32)

    else:
        meas_cov = compute_measurement_covariances(rad_meas, rad_meas_rms[:,0], rad_meas_rms[:,1], rad_meas_rms[:,2], rad_meas_rms[:,3])
        sampling_cov = meas_cov

        if num_samples > 0:
            xyvxvy_samples, cov_samples = generate_samples(rad_meas, meas_cov, sampling_cov, num_samples)
            rad_meas = np.concatenate( [rad_meas, xyvxvy_samples], axis=0 )
            meas_cov = np.concatenate( [meas_cov, cov_samples] , axis=0 )
            are_samples = np.concatenate( [are_samples, np.ones((xyvxvy_samples.shape[0], ), dtype=np.bool8)], axis=0 )

    return rad_meas, meas_cov, are_samples

# =====================================================================================================================