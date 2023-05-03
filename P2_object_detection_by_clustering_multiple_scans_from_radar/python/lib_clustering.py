# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Clustering
# =====================================================================================================================

import config
import numpy as np

# =====================================================================================================================

def compute_cov_ellipse(mu, cov, chi_sq, n_points):

    # eigenvalue decomposition
    d, v = np.linalg.eig(cov)

    # idx corresponding to largest and smallest eig values
    largest_eigval_idx = np.where(d == np.max(d))[0]
    largest_eigval_idx = np.reshape(largest_eigval_idx, -1)
    if largest_eigval_idx[0] == 0:
        largest_eigval_idx = 0
        smallest_eigval_idx = 1
    elif largest_eigval_idx[0] == 1:
        largest_eigval_idx = 1
        smallest_eigval_idx = 0

    # ellipse axis half lengths
    a = chi_sq * np.sqrt(d[largest_eigval_idx]) 
    b = chi_sq * np.sqrt(d[smallest_eigval_idx])

    # compute ellipse orientation
    if (v[largest_eigval_idx, 0] == 0): theta = np.pi / 2
    else: theta = np.arctan2(v[largest_eigval_idx, 1], v[largest_eigval_idx, 0])

    # genetare theta and points(x,y)
    th = np.linspace(0, 2 * np.pi, n_points)
    px = a * np.cos(th)
    py = b * np.sin(th)
    points = np.stack((px,py), axis=-1)

    # Rotate by theta and translate by mu
    R = np.array([ [ np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)] ], dtype=np.float64)
    T = mu
    points = points @ R.transpose(1,0) + T
    return points, mu

# =====================================================================================================================

def compute_l2_norm(meas_vector_i, meas_vector_js):
    l2 = np.expand_dims(np.expand_dims(meas_vector_i, axis=0) - meas_vector_js, axis=-1)
    l2 = l2.transpose(0,2,1) @ l2
    return l2

def compute_mahalanobis_dist(meas_vector_i, meas_vector_js, meas_cov_i, meas_cov_js):
    meas_cov_ij = np.expand_dims(meas_cov_i, axis=0) + meas_cov_js
    error = np.expand_dims(np.expand_dims(meas_vector_i, axis=0) - meas_vector_js, axis=-1)
    m_dist = error.transpose(0,2,1) @ np.linalg.inv( meas_cov_ij ) @ error
    return m_dist

def compute_l1_norm(meas_vector_i, meas_vector_js):
    l1 = np.expand_dims(meas_vector_i, axis=0) - meas_vector_js
    l1 = np.abs(l1)
    return l1

# =====================================================================================================================

def compute_distance1(meas_vector_i, meas_vector_js, eps_pxpy, eps_vxvy):
    condition1 = compute_l2_norm(meas_vector_i[:2], meas_vector_js[:, :2]) <= eps_pxpy
    condition2 = compute_l2_norm(meas_vector_i[2:], meas_vector_js[:, 2:]) <= eps_vxvy
    condition = np.logical_and(condition1, condition2)
    return np.reshape(condition, -1)

def compute_distance2(meas_vector_i, meas_vector_js, eps):
    condition = compute_l2_norm(meas_vector_i, meas_vector_js) <= eps
    return np.reshape(condition, -1)

def compute_distance3(meas_vector_i, meas_vector_js, meas_cov_i, meas_cov_js, eps_pxpy, eps_vxvy):
    condition1 = compute_mahalanobis_dist(meas_vector_i[:2], meas_vector_js[:, :2], meas_cov_i[:2, :2], meas_cov_js[:, :2, :2]) <= eps_pxpy
    condition2 = compute_mahalanobis_dist(meas_vector_i[2:], meas_vector_js[:, 2:], meas_cov_i[2:, 2:], meas_cov_js[:, 2:, 2:]) <= eps_vxvy
    condition = np.logical_and(condition1, condition2)
    return np.reshape(condition, -1)

def compute_distance4(meas_vector_i, meas_vector_js, meas_cov_i, meas_cov_js, eps):
    condition = compute_mahalanobis_dist(meas_vector_i, meas_vector_js, meas_cov_i, meas_cov_js) <= eps
    return np.reshape(condition, -1)

# =====================================================================================================================

def compute_cluster_sample_mean_and_cov(meas_vector):

    meas_mu = np.sum(meas_vector, axis=0) / meas_vector.shape[0]
    meas_cov = config.R
    meas_shape = config.cluster_shape

    if meas_vector.shape[0] > 1:
        # error = np.expand_dims((meas_mu[:2] - meas_vector[:, :2]), axis=-1)
        error = np.expand_dims((meas_mu[:4] - meas_vector[:, :4]), axis=-1)
        sigma = error @ error.transpose(0, 2, 1)
        sigma = np.sum(sigma, axis=0) / ( meas_vector.shape[0] - 1 )
        # meas_shape = sigma + meas_shape
        meas_shape = sigma + meas_cov

    return meas_mu, meas_cov, meas_shape

# =====================================================================================================================

class DBSCAN:

    def __init__(
        self, 
        max_num_measurements,
        eps,
        eps_num_pts):

        self.eps = eps
        self.eps_num_pts = eps_num_pts

        self.num_clusters = 0
        self.num_noise_pts = 0
        self.meas_vector = np.zeros((max_num_measurements, config.meas_dim), dtype=np.float32)
        self.meas_covariance = np.zeros((max_num_measurements, config.meas_dim, config.meas_dim), dtype=np.float32)
        self.shape_covariance = np.zeros((max_num_measurements, 4, 4), dtype=np.float32)
        self.meas_dynamic_status = np.zeros((max_num_measurements, ), dtype=np.bool8)
        self.meas_to_cluster_id = -1 + np.zeros((max_num_measurements, ), dtype=np.int16)
        self.num_measurements = np.zeros((max_num_measurements, ), dtype=np.int16)

        # init internal data
        self.cluster_member = np.zeros((max_num_measurements, ), dtype=np.uint16)
        self.cluster_core_member = np.zeros((max_num_measurements, ), dtype=np.uint16)
        self.assignmemnt_matrix = np.zeros((max_num_measurements, max_num_measurements), dtype=np.bool8)
        self.is_meas_core = np.zeros((max_num_measurements, ), dtype=np.bool8)


    def reinitialize(self):
        self.meas_to_cluster_id.fill(-1)
        self.num_clusters = 0


    def dbscan(
        self, 
        meas_vector,
        meas_covariance,
        meas_dynamic_status):
        """ dbscan clustering of radar measurements.
        Inputs: meas_vector - array of vectors of shape (n, 4), each attr is (px,py,vx,vy)
              : meas_dynamic_status - a boolean vector indicating if the measurement is dynamic or stationary
        Outputs: 
        """
        self.reinitialize()
        num_clstr_mem = 0
        num_clstr_core_mem = 0
        cluster_id = 0
        
        # extract data
        num_meas = meas_vector.shape[0]

        # identify which points are core points
        for i in range(num_meas):
            condition = compute_distance4(meas_vector[i], meas_vector[i:], meas_covariance[i], meas_covariance[i:], self.eps)
            self.assignmemnt_matrix[i, i:num_meas] = condition
            self.assignmemnt_matrix[i:num_meas, i] = condition
            self.assignmemnt_matrix[i,  i] = False
            self.is_meas_core[i] = np.sum(self.assignmemnt_matrix[i, :num_meas], dtype=np.int16) >= self.eps_num_pts

        # start clustering from the core points
        for m in range(num_meas):
            if ( self.is_meas_core[m] and self.meas_to_cluster_id[m] == -1 ):             

                self.cluster_member[num_clstr_mem] = m
                self.cluster_core_member[num_clstr_core_mem] = m
                num_clstr_mem += 1
                num_clstr_core_mem += 1
                self.meas_to_cluster_id[m] = cluster_id

                n = 0
                while n < num_clstr_core_mem:
                    i = self.cluster_core_member[n]                                  

                    for j in range(num_meas):
                        condition = ( self.meas_to_cluster_id[j] == -1 and self.assignmemnt_matrix[i, j] )
                        condition1 = condition and self.is_meas_core[j]
                        condition2 = condition and np.logical_not(self.is_meas_core[j])

                        if condition1:
                            self.cluster_member[num_clstr_mem] = j
                            self.cluster_core_member[num_clstr_core_mem] = j
                            num_clstr_mem += 1
                            num_clstr_core_mem += 1
                            self.meas_to_cluster_id[j] = cluster_id

                        elif condition2:
                            self.cluster_member[num_clstr_mem] = j
                            num_clstr_mem += 1
                            self.meas_to_cluster_id[j] = cluster_id

                    n += 1
             
                # compute cluster mean and covariance
                meas_idx = self.cluster_member[:num_clstr_mem]
                clstr_mu, clstr_cov, clstr_shape = compute_cluster_sample_mean_and_cov(meas_vector[meas_idx, :])

                # update the cluster data
                self.num_clusters = cluster_id + 1
                self.meas_vector[cluster_id, :] = clstr_mu
                self.meas_covariance[cluster_id, :, :] = clstr_cov
                self.shape_covariance[cluster_id, :, :] = clstr_shape
                self.meas_dynamic_status[cluster_id] = meas_dynamic_status[meas_idx][0]
                self.num_measurements[cluster_id] = num_clstr_mem

                cluster_id += 1
                num_clstr_core_mem = 0
                num_clstr_mem = 0

        self.num_noise_pts = np.sum(self.meas_to_cluster_id[:num_meas] == -1, dtype=np.int16)

# =====================================================================================================================

def concatenate_clusters(rad_clusters):

    num_clusters = 0
    meas_vector = []
    meas_covariance = []
    shape_covariance = []
    num_measurements = []
    sensor_id = []

    for rad_id in range(len(rad_clusters)):
        num_clusters += rad_clusters[rad_id].num_clusters

        if rad_clusters[rad_id].num_clusters > 0:
            meas_vector.append(rad_clusters[rad_id].meas_vector[:rad_clusters[rad_id].num_clusters, :])
            meas_covariance.append(rad_clusters[rad_id].meas_covariance[:rad_clusters[rad_id].num_clusters, :, :])
            shape_covariance.append(rad_clusters[rad_id].shape_covariance[:rad_clusters[rad_id].num_clusters, :, :])
            num_measurements.append(rad_clusters[rad_id].num_measurements[:rad_clusters[rad_id].num_clusters])
            sensor_id.append(np.repeat(rad_id, rad_clusters[rad_id].num_clusters).astype(np.int16))

    if num_clusters > 0:
        meas_vector = np.concatenate(meas_vector, axis=0)
        meas_covariance = np.concatenate(meas_covariance, axis=0)
        shape_covariance = np.concatenate(shape_covariance, axis=0)
        num_measurements = np.concatenate(num_measurements, axis=0)
        sensor_id = np.concatenate(sensor_id, axis=0)

    return num_clusters, meas_vector, meas_covariance, shape_covariance, num_measurements, sensor_id

# =====================================================================================================================

def merge_clusters(mu, sigma, num_samples):
    # init
    total_num_samples = np.sum(num_samples)
    num_clusters = num_samples.shape[0]
    # compute mean
    wt = np.expand_dims(num_samples / total_num_samples, axis=-1)
    mean = np.sum(wt * mu, axis=0)
    # compute covariance 
    wt1 = np.expand_dims((num_samples - 1) / (total_num_samples - 1), axis=(1,-1))
    wt2 = np.expand_dims(num_samples / (total_num_samples - 1), axis=(1,-1))
    error = np.expand_dims(mu - mean, axis=-1)
    covariance = np.sum(wt1 * sigma, axis=0) + np.sum(wt2 * error @ error.transpose(0,2,1), axis=0)
    return mean, covariance

# =====================================================================================================================