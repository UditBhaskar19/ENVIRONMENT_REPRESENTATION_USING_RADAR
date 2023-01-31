# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Library for Log-Odds computation
# =====================================================================================================================

import numpy as np
import lib_const
from lib_grid import (
    select_points_within_grid_boundary,
    select_cells_within_the_grid_boundary
)

# =====================================================================================================================

def compute_prob_from_log_odds(log_odds):
    """ Compute probability from log odds """
    log_odds = np.clip(log_odds, -23.0, 23.0)   # to pervent overflow
    prob = 1 - ( 1 / (1 + np.exp(log_odds)) )
    return prob


def compute_log_odds_from_prob(prob):
    """ compute log odds from probability """
    log_odds = np.log((prob + lib_const.eps) / (1.0 - prob + lib_const.eps))
    return log_odds


def inflate_prob_and_compute_log_odds(prob):
    """ translates and scales probabilities such that it always lies within [0.5, 1] """
    prob = 0.5 + 0.5 * prob
    log_odds = compute_log_odds_from_prob(prob)
    return log_odds

# =====================================================================================================================

def compute_covariance_linearly_varying(
    meas_x, 
    meas_y, 
    res_map_x, 
    res_map_y):

    meas_dx = res_map_x.compute_resolution_linear(np.abs(meas_x))
    meas_dy = res_map_y.compute_resolution_linear(np.abs(meas_y))
    meas_cov = np.repeat(np.expand_dims(np.eye(2), axis=0), meas_dx.shape[0], axis=0)
    meas_res = np.stack([meas_dx, meas_dy], axis=-1)
    meas_res = np.repeat(np.expand_dims(meas_res, axis=-1), 2, axis=-1)
    meas_cov = meas_cov * meas_res
    return meas_cov


def compute_covariance_exponentially_varying(
    meas_x, 
    meas_y, 
    res_map_x, 
    res_map_y):

    meas_dx = res_map_x.compute_resolution_exponential(np.abs(meas_x))
    meas_dy = res_map_y.compute_resolution_exponential(np.abs(meas_y))
    meas_cov = np.repeat(np.expand_dims(np.eye(2), axis=0), meas_dx.shape[0], axis=0)
    meas_res = np.stack([meas_dx, meas_dy], axis=-1)
    meas_res = np.repeat(np.expand_dims(meas_res, axis=-1), 2, axis=-1)
    meas_cov = meas_cov * meas_res
    return meas_cov


def compute_covariance_constant(
    meas, 
    sigma):

    meas_cov = np.repeat(np.expand_dims(sigma*np.eye(2), axis=0), meas.shape[0], axis=0)
    return meas_cov

# =====================================================================================================================

def generate_samples(
    meas_xcoord,
    meas_ycoord, 
    meas_cov, 
    num_samples):
    """ Generating samples and computing the weights of the samples """
    samples = []
    weights = []
    mu = np.stack([meas_xcoord, meas_ycoord], axis=-1)
    meas_cov_inv = np.linalg.inv(meas_cov)

    for i in range(meas_xcoord.shape[0]):
        particles = np.random.multivariate_normal(mu[i], meas_cov[i], num_samples)
        wt = np.expand_dims(mu[i], axis=(0,2)) - np.expand_dims(particles, axis=-1)
        wt = wt.transpose(0,2,1) @ np.expand_dims(meas_cov_inv[i], axis=0) @ wt
        wt = np.exp( -0.5 * wt )
        samples.append(particles)
        weights.append(np.reshape(wt, -1))

    samples.append(mu)
    weights.append( np.ones((meas_xcoord.shape[0] , ), dtype=np.float32) )
    samples = np.concatenate( samples, axis=0 )
    weights = np.concatenate( weights , axis=0 )
    return samples[:,0], samples[:,1], weights

# =====================================================================================================================

def compute_meas_log_likelihood(
    meas_x, 
    meas_y, 
    sigma, 
    num_samples, 
    grid):
    """ create dense measurement log-likelihood by sampling more particles """
    # generate more samples by random sampling
    meas_cov = compute_covariance_constant(meas_x, sigma)
    meas_x, meas_y, weights = generate_samples(meas_x, meas_y, meas_cov, num_samples)

    # select meas which are within the pre-defined grid
    sel_flag = select_points_within_grid_boundary(
        meas_x, meas_y, 
        grid.min_x, grid.max_x,
        grid.min_y, grid.max_y)

    # extracte meas 
    meas_x_roi = meas_x[sel_flag]
    meas_y_roi = meas_y[sel_flag]
    weights_roi = weights[sel_flag]

    # sort in the increasing order of weights
    idx = np.argsort(weights_roi)
    weights_roi = weights_roi[idx]
    meas_x_roi = meas_x_roi[idx]
    meas_y_roi = meas_y_roi[idx]

    # compute cell scalar ids, remove duplicate copies and compute loglikelihood
    condition = np.zeros((grid.num_cells, ), dtype=np.bool8)
    cell_ids = np.arange(grid.num_cells, dtype=np.uint32)
    prob_vals = np.zeros((grid.num_cells, ), dtype=np.float32)
    xcoords = np.zeros((grid.num_cells, ), dtype=np.float32) 
    ycoords = np.zeros((grid.num_cells, ), dtype=np.float32)

    cellids = grid.compute_scalar_ids_from_xy_coordinates(meas_x_roi, meas_y_roi)
    condition[cellids] = True 
    prob_vals[cellids] = weights_roi
    # cell_ids[cellids] = cellids
    xcoords[cellids] = meas_x_roi
    ycoords[cellids] = meas_y_roi

    return (
        prob_vals[condition], 
        cell_ids[condition],
        xcoords[condition],
        ycoords[condition] )

# =====================================================================================================================

class kernel:
    """ This is used to approximate the n-sigma covariance of the 
    measurements as a rectangular shape to create the log-odds map
    """
    def __init__(
        self,
        num_rows, 
        num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.kernel_size = num_rows * num_cols
        self.kernel_idx = self.create_kernel_idx()

    def create_kernel_idx(self):
        col_idx = np.arange(self.num_cols) - self.num_cols // 2
        row_idx = np.arange(self.num_rows) - self.num_rows // 2
        col_idx, row_idx = np.meshgrid(col_idx, row_idx)
        kernel_idx = np.reshape(np.stack([row_idx, col_idx], axis=-1), (-1, 2))
        return kernel_idx

# =====================================================================================================================

class res_map:
    """ The functions here inflates the covaiance linearly 
    or exponentially using some heuristics """
    def __init__(
        self,
        min_x,
        max_x,
        min_dx,
        max_dx):

        self.min_x = min_x
        self.max_x = max_x
        self.min_dx = min_dx
        self.max_dx = max_dx
        self.A = np.log( max_dx / (min_dx + lib_const.eps) )  / ( max_x - min_x ) 
        self.B = np.exp( np.log(min_dx + lib_const.eps) - self.A*min_x )

    def compute_resolution_exponential(self, x):
        dx = self.B * np.exp( self.A * x )
        dx = np.clip(dx, self.min_dx, self.max_dx)
        return dx

    def compute_resolution_linear(self, x):
        dx = ( x - self.min_x ) * ( self.max_dx - self.min_dx ) / ( self.max_x - self.min_x ) + self.min_dx
        dx = np.clip(dx, self.min_dx, self.max_dx)
        return dx

# =====================================================================================================================

def compute_meas_log_likelihood_deterministic_sampling(
    meas_x, 
    meas_y, 
    sigma, 
    kernel, 
    grid):
    """ compute measurement log-likelihood image by generating more samples from regularily spaced grid """
    # select meas which are with-in the pre-defined grid
    sel_flag = select_points_within_grid_boundary(
        meas_x, meas_y, 
        grid.min_x, grid.max_x,
        grid.min_y, grid.max_y)

    # create meas stack
    meas_x_roi = meas_x[sel_flag]
    meas_y_roi = meas_y[sel_flag]
    meas = np.stack([meas_x_roi, meas_y_roi], axis=-1)
    meas = np.expand_dims(meas, axis=(1, -1))

    # compute grid cell index and cell coordinates
    cellxyids = np.stack(grid.compute_xy_ids_from_xy_coordinates(meas_x_roi, meas_y_roi), axis=-1)
    cellxyids = np.expand_dims(cellxyids, axis=(1, -1))
    kernelidx = np.expand_dims(kernel.kernel_idx, axis=(0, -1))
    cellxyids = cellxyids + kernelidx

    # compute cell coordinates
    cellxyids = np.reshape(np.squeeze(cellxyids, axis=-1), (-1, 2))
    cellcoord = np.stack(grid.compute_xy_coordinates_from_xy_ids(cellxyids[:,0], cellxyids[:,1]), axis=-1)
    cellcoord = np.reshape(cellcoord, (meas.shape[0], kernel.kernel_size, 2, 1))

    # compute meas covariance
    meas_cov = compute_covariance_constant(meas_x_roi, sigma)
    meas_cov = np.expand_dims(meas_cov, axis=1)

    # compute probability
    error = meas - cellcoord
    error = error.transpose(0,1,3,2) @ np.linalg.inv(meas_cov) @ error
    prob = np.exp( -0.5 * error )
    prob = np.reshape( np.squeeze(prob, axis=(2,3)), -1 )

    # select cells that are within the defined grid
    sel_flag = select_cells_within_the_grid_boundary(
        cellxyids[:, 0], cellxyids[:, 1], grid.num_cells_x, grid.num_cells_y)
    prob = prob[sel_flag]
    cellxyids = cellxyids[sel_flag]

    # sort probabilities in ascending order
    idx = np.argsort(prob)
    prob = prob[idx]
    cellxid = cellxyids[idx, 0]
    cellyid = cellxyids[idx, 1]
    cellids = grid.compute_scalar_ids_from_xy_ids(cellxid, cellyid)
    
    # compute cell scalar ids, remove duplicate copies and compute loglikelihood
    condition = np.zeros((grid.num_cells, ), dtype=np.bool8)
    cell_ids = np.arange(grid.num_cells, dtype=np.uint32)
    prob_vals = np.zeros((grid.num_cells, ), dtype=np.float32)
    condition[cellids] = True 
    prob_vals[cellids] = prob

    # extract the unique cell copies and compute the cell coordinates
    prob_vals = prob_vals[condition]
    cell_ids = cell_ids[condition]
    xcoords, ycoords = grid.compute_xy_coordinates_from_scalar_ids(cell_ids)

    return (
        prob_vals, 
        cell_ids,
        xcoords,
        ycoords )



