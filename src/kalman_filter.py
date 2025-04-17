# modified from: https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
# implemented vectorized versions of functions for faster speed

import numpy as np


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.zeros((8, 8))
        np.fill_diagonal(covariance, np.square(std))
        
        return mean, covariance
    
    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (vectorized version).
        Parameters
        ----------
        mean : ndarray
            The mean vectors (Nx8) of the object states at the previous
            time step.
        covariance : ndarray
            The covariance matrices (Nx8x8) of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vectors (Nx8) and covariance matrices (Nx8x8) of the predicted
            states. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        motion_cov = np.zeros((mean.shape[0], 8, 8))
        for i in range(len(mean)):
            np.fill_diagonal(motion_cov[i], sqr[i])
        
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.dot(self._motion_mat @ covariance, self._motion_mat.T) + motion_cov
        
        return mean, covariance
    
    def multi_project(self, mean, covariance, use_nsa=False, nsa_use_square=True, nsa_scale_factor=1.0, confidence=.0):
        """Project state distribution to measurement space (vectorized version).
        Parameters
        ----------
        mean : ndarray
            The states' mean vectors (Nx8).
        covariance : ndarray
            The states' covariance matrices (Nx8x8).
        confidence : float
            The confidences (N) of the measurements.
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected means (Nx4) and covariance matrices (Nx4x4) of the given state
            estimates.
        """
        std = np.array([
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-1 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]])
        
        if use_nsa:
            if nsa_use_square:
                std = (1 - confidence) * nsa_scale_factor * std
            else:
                std = np.sqrt((1 - confidence) * nsa_scale_factor) * std
        
        innovation_cov = np.zeros((mean.shape[0], 4, 4))
        for i in range(len(mean)):
            np.fill_diagonal(innovation_cov[i], np.square(std[:, i].T))
        
        mean = np.dot(mean, self._update_mat.T)
        
        left = np.dot(self._update_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._update_mat.T)
        return mean, covariance + innovation_cov
    
    def multi_update(self, mean, covariance, measurement, use_nsa=False, nsa_use_square=True, nsa_scale_factor=1.0, confidence=.0, use_cw=False, cw_score_thresh=0.0, cw_scale_factor=1.0):
        """Run Kalman filter correction step (vectorized version).
        Parameters
        ----------
        mean : ndarray
            The predicted states' mean vectors (Nx8).
        covariance : ndarray
            The states' covariance matrices (Nx8x8).
        measurement : ndarray
            The 4 dimensional measurement vectors (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box (Nx4).
        confidence : ndarray
            The confidences of the measurements (N).
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution (Nx8), (Nx8x8).
        """
        projected_mean, projected_cov = self.multi_project(mean, covariance, use_nsa=use_nsa, nsa_use_square=nsa_use_square, nsa_scale_factor=nsa_scale_factor, confidence=confidence)
        
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        
        if use_cw:
            det_has_low_conf = confidence < cw_score_thresh
            if np.any(det_has_low_conf):
                delta = measurement[det_has_low_conf] - mean[det_has_low_conf, :4]
                measurement[det_has_low_conf, :4] = mean[det_has_low_conf, :4] + delta * confidence[det_has_low_conf, None] * cw_scale_factor
        
        innovation = measurement - projected_mean
        
        new_mean = mean + np.matmul(kalman_gain, innovation[..., None])[..., 0]
        left = np.matmul(kalman_gain, projected_cov)
        full = np.matmul(left, kalman_gain.transpose((0, 2, 1)))
        new_covariance = covariance - full
        
        return new_mean, new_covariance
