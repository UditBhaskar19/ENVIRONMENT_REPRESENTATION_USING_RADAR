# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validate merge cluster
# =====================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from lib_clustering import compute_cov_ellipse, merge_clusters


translation_vector1 = np.array([-4, 4], dtype=np.float32)
scaling_factor1 = 1.25
rotation_angle1 = ( np.pi / 180 ) * 20
rotation_matrix1 = np.array([[np.cos(rotation_angle1),  -np.sin(rotation_angle1)], [np.sin(rotation_angle1),   np.cos(rotation_angle1)]], dtype=np.float32)
transformation_mat1 = scaling_factor1 * rotation_matrix1


translation_vector2 = np.array([-5, 0], dtype=np.float32)
scaling_factor2 = 0.5
rotation_angle2 = ( np.pi / 180 ) * -5
rotation_matrix2 = np.array([[np.cos(rotation_angle2),  -np.sin(rotation_angle2)], [np.sin(rotation_angle2),   np.cos(rotation_angle2)]], dtype=np.float32)
transformation_mat2 = scaling_factor2 * rotation_matrix2


mean0 = np.array([0, 0], dtype=np.float32)
covariance0 = np.array([[3, 0], [0, 3]], dtype=np.float32)
num_samples0 = 500


mean1 = mean0 + translation_vector1
covariance1 = np.array([[5, 0], [0, 2]], dtype=np.float32)
covariance1 = transformation_mat1 @ covariance1 @ transformation_mat1.transpose()
num_samples1 = 500


mean2 = mean0 + translation_vector2
covariance2 = np.array([[5, 0], [0, 2]], dtype=np.float32)
covariance2 = transformation_mat2 @ covariance2 @ transformation_mat2.transpose()
num_samples2 = 200


samples0 = np.random.multivariate_normal(mean0, covariance0, size=num_samples0)
samples1 = np.random.multivariate_normal(mean1, covariance1, size=num_samples1)
samples2 = np.random.multivariate_normal(mean2, covariance2, size=num_samples2)


samples_stack = np.array([num_samples0, num_samples1, num_samples2])
mean_stack = np.stack([mean0, mean1, mean2], axis=0)
covariance_stack = np.stack([covariance0, covariance1, covariance2], axis=0)
mean, covariance = merge_clusters(mean_stack, covariance_stack, samples_stack)


boundary_points0, _ = compute_cov_ellipse(mean0, covariance0, 3, 2000)
boundary_points1, _ = compute_cov_ellipse(mean1, covariance1, 3, 2000)
boundary_points2, _ = compute_cov_ellipse(mean2, covariance2, 3, 2000)
boundary_points,  _ = compute_cov_ellipse(mean,  covariance,  3, 2000)



fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_aspect('equal')
ax[0].set_xlabel('x (m)')
ax[0].set_ylabel('y (m)')

ax[0].scatter(samples0[:, 0], samples0[:, 1], s=10, color='magenta', marker='.', label='cluster 1 $3\sigma$ cov boundary')
ax[0].scatter(samples1[:, 0], samples1[:, 1], s=10, color='green', marker='.', label='cluster 2 $3\sigma$ cov boundary')
ax[0].scatter(samples2[:, 0], samples2[:, 1], s=10, color='red', marker='.', label='cluster 3 $3\sigma$ cov boundary')

ax[0].scatter(boundary_points0[:, 0], boundary_points0[:, 1], s=3, color='magenta', marker='.')
ax[0].scatter(boundary_points1[:, 0], boundary_points1[:, 1], s=3, color='green', marker='.')
ax[0].scatter(boundary_points2[:, 0], boundary_points2[:, 1], s=3, color='red', marker='.')

ax[0].scatter(mean0[0], mean0[1], s=200, color='black', marker='x', linewidth=3)
ax[0].scatter(mean1[0], mean1[1], s=200, color='black', marker='x', linewidth=3)
ax[0].scatter(mean2[0], mean2[1], s=200, color='black', marker='x', linewidth=3)

ax[0].legend(loc='lower left')


ax[1].set_aspect('equal')
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('y (m)')

ax[1].scatter(samples0[:, 0], samples0[:, 1], s=10, color='magenta', marker='.')
ax[1].scatter(samples1[:, 0], samples1[:, 1], s=10, color='green', marker='.')
ax[1].scatter(samples2[:, 0], samples2[:, 1], s=10, color='red', marker='.')

ax[1].scatter(boundary_points[:, 0], boundary_points[:, 1], s=0.3, color='black', marker='.')
ax[1].scatter(mean[0], mean[1], s=200, color='black', marker='x', linewidth=3, label='merged cluster')

ax[1].legend(loc='lower left')
plt.show()