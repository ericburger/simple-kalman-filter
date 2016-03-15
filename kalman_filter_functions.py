#####################################################################################
#
#  Copyright (c) 2016 Eric Burger
# 
#  MIT License (MIT)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy 
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
#  copies of the Software, and to permit persons to whom the Software is 
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
#  SOFTWARE.
#
#####################################################################################

import numpy as np
from scipy import linalg


'''

Kalman Filter

'''
def kf_estimate_x( A_matrix, B_matrix, x_mean_previous, Q_x_cov_previous, u_input, Q_v_cov ):
    
    x_mean_priori = np.dot( A_matrix, x_mean_previous ) + np.dot( B_matrix, u_input )
    
    Q_x_cov_priori = np.dot(A_matrix, np.dot(Q_x_cov_previous, A_matrix.T)) + Q_v_cov

    return x_mean_priori, Q_x_cov_priori
    
def kf_estimate_y( C_matrix, D_matrix, x_mean_priori, Q_x_cov_priori, u_input, Q_w_cov ):
    
    y_mean = np.dot( C_matrix, x_mean_priori ) + np.dot( D_matrix, u_input )
    
    Q_y_cov = np.dot(C_matrix, np.dot(Q_x_cov_priori, C_matrix.T)) + Q_w_cov
    
    return y_mean, Q_y_cov

def kf_correct( x_mean_priori, Q_x_cov_priori, y_mean, Q_y_cov, C_matrix, y_k ):

    # calculate Kalman gain
    K = np.dot( Q_x_cov_priori, np.dot( C_matrix.T, linalg.pinv(Q_y_cov) ) )

    # correct mu, sigma
    residual = y_k - y_mean
    x_mean_posterior = x_mean_priori + K.dot( residual )
    Q_x_cov_posterior = Q_x_cov_priori - K.dot( np.dot(Q_y_cov, K.T))

    return x_mean_posterior, Q_x_cov_posterior, residual

def kf_predict_update( x_mean_previous, Q_x_cov_previous, u_input, A_matrix, B_matrix, C_matrix, D_matrix, Q_v_cov, Q_w_cov, y_k ):

    x_mean_priori, Q_x_cov_priori = kf_estimate_x( A_matrix, B_matrix, x_mean_previous, Q_x_cov_previous, u_input, Q_v_cov )

    y_mean, Q_y_cov = kf_estimate_y( C_matrix, D_matrix, x_mean_priori, Q_x_cov_priori, u_input, Q_w_cov )

    x_mean_posterior, Q_x_cov_posterior, residual = kf_correct( x_mean_priori, Q_x_cov_priori, y_mean, Q_y_cov, C_matrix, y_k )

    return x_mean_posterior, Q_x_cov_posterior, residual


'''

Unscented Transform

'''

# Generature Sigma Points
def ut_sigma(mean, cov, alpha=None, beta=None, kappa=None):
    # mean: state mean, n_dim array
    # cov: state covariance, n_dim by n_dim array
    n_dim = len(mean)
    mean = np.asarray(np.atleast_2d(mean), dtype=float, order=None)

    if alpha is None:
      alpha = 1.0
    if beta is None:
      beta = 0.0
    if kappa is None:
      kappa = 3.0 - n_dim

    # compute sqrt(cov)
    sqrt_cov = linalg.cholesky(cov).T
    #sqrt_cov = linalg.sqrtm(cov)

    # Calculate scaling factor for all off-center points
    lamda = (alpha * alpha) * (n_dim + kappa) - n_dim
    c = n_dim + lamda

    # calculate the sigma points; that is,
    #   mean
    #   mean + each column of sqrt_cov * sqrt(c)
    #   mean - each column of sqrt_cov * sqrt(c)
    # Each column of points is one of these.
    points = np.tile(mean.T, (1, 2 * n_dim + 1))
    points[:, 1:(n_dim + 1)] += sqrt_cov * np.sqrt(c)
    points[:, (n_dim + 1):] -= sqrt_cov * np.sqrt(c)

    # Calculate weights
    weights_mean = np.ones(2 * n_dim + 1)
    weights_mean[0] = lamda / c
    weights_mean[1:] = 0.5 / c
    weights_cov = np.copy(weights_mean)
    weights_cov[0] = lamda / c + (1 - alpha * alpha + beta)

    return points.T, weights_mean, weights_cov


def ut_mean_and_cov(sigma_points, weights_mean, weights_cov, additive_noise_cov=None):
    mean = sigma_points.T.dot(weights_mean)
    points_diff = sigma_points.T - mean[:, np.newaxis]
    cov = points_diff.dot(np.diag(weights_cov)).dot(points_diff.T)
    if additive_noise_cov is not None:
        cov = cov + additive_noise_cov
    return mean, cov    

'''

Unscented Kalman Filter

'''

# Generature Sigma Points
def ukf_additive_sigma_points( priori_state_mean, priori_state_cov ):
    return ut_sigma( priori_state_mean, priori_state_cov )
    
def ukf_additive_estimate_x( F_func, sigma_x, u_input, Q_v_cov, weights_mean, weights_cov ):
    n_points, n_dim_state = sigma_x.shape
    X_sigma_points_priori =  [F_func(sigma_x[i],u_input) for i in range(n_points)]
    X_sigma_points_priori = np.vstack(X_sigma_points_priori)
    x_mean_priori, Q_x_cov_priori = ut_mean_and_cov(X_sigma_points_priori, weights_mean, weights_cov, Q_v_cov)
    return x_mean_priori, Q_x_cov_priori, X_sigma_points_priori
    
def ukf_additive_estimate_y( H_func, sigma_x_priori, u_input, Q_w_cov, weights_mean, weights_cov ):
    n_points, n_dim_state = sigma_x_priori.shape
    Y_sigma_points =  [H_func(sigma_x_priori[i],u_input) for i in range(n_points)]
    Y_sigma_points = np.vstack(Y_sigma_points)
    y_mean, Q_y_cov = ut_mean_and_cov(Y_sigma_points, weights_mean, weights_cov, Q_w_cov)
    return y_mean, Q_y_cov, Y_sigma_points

def ukf_additive_correct( x_mean_priori, Q_x_cov_priori, X_sigma_points_priori, y_mean, Q_y_cov, Y_sigma_points, weights_mean, weights_cov, y_k ):
    # calculate the cross covariance
    Q_xy = (
        ((X_sigma_points_priori - x_mean_priori).T)
        .dot(np.diag(weights_cov))
        .dot(Y_sigma_points - y_mean)
    )

    # calculate Kalman gain
    K = Q_xy.dot(linalg.pinv(Q_y_cov))

    # correct mu, sigma
    residual = y_k - y_mean
    x_mean_posterior = x_mean_priori + K.dot(residual)
    Q_x_cov_posterior = Q_x_cov_priori - K.dot(Q_xy.T)

    
    return x_mean_posterior, Q_x_cov_posterior, residual

def ukf_additive_predict_update( x_mean_previous, Q_x_cov_previous, u_input, F_transition_function, H_observation_function, Q_v_cov, Q_w_cov, y_k ):

    S_x_sigma_points, weights_mean, weights_cov = ukf_additive_sigma_points( x_mean_previous, Q_x_cov_previous )

    x_mean_priori, Q_x_cov_priori, X_sigma_points_priori = ukf_additive_estimate_x( F_transition_function, S_x_sigma_points, u_input, Q_v_cov, weights_mean, weights_cov  )

    y_mean, Q_y_cov, Y_sigma_points = ukf_additive_estimate_y( H_observation_function, X_sigma_points_priori, u_input, Q_w_cov, weights_mean, weights_cov )

    x_mean_posterior, Q_x_cov_posterior, residual = ukf_additive_correct( x_mean_priori, Q_x_cov_priori, X_sigma_points_priori, y_mean, Q_y_cov, Y_sigma_points, weights_mean, weights_cov, y_k )

    return x_mean_posterior, Q_x_cov_posterior, residual
    

def constrained_ukf_additive_predict_update( x_mean_previous, Q_x_cov_previous, u_input, F_transition_function, H_observation_function, Q_v_cov, Q_w_cov, y_k, x_upper=None, x_lower=None ):
    
    S_x_sigma_points, weights_mean, weights_cov = ukf_additive_sigma_points( x_mean_previous, Q_x_cov_previous )
    
    for i in range(len(S_x_sigma_points)):
        #print S_x_sigma_points[i]
        #print np.minimum( S_x_sigma_points[i], x_upper )
        if x_upper is not None:
            S_x_sigma_points[i] = np.minimum( S_x_sigma_points[i], x_upper )
        if x_lower is not None:
            S_x_sigma_points[i] = np.maximum( S_x_sigma_points[i], x_lower )
    
    x_mean_priori, Q_x_cov_priori, X_sigma_points_priori = ukf_additive_estimate_x( F_transition_function, S_x_sigma_points, u_input, Q_v_cov, weights_mean, weights_cov  )

    for i in range(len(X_sigma_points_priori)):
        if x_upper is not None:
            X_sigma_points_priori[i] = np.minimum( X_sigma_points_priori[i], x_upper )
        if x_lower is not None:
            X_sigma_points_priori[i] = np.maximum( X_sigma_points_priori[i], x_lower )
        
    y_mean, Q_y_cov, Y_sigma_points = ukf_additive_estimate_y( H_observation_function, X_sigma_points_priori, u_input, Q_w_cov, weights_mean, weights_cov )

    x_mean_posterior, Q_x_cov_posterior, residual = ukf_additive_correct( x_mean_priori, Q_x_cov_priori, X_sigma_points_priori, y_mean, Q_y_cov, Y_sigma_points, weights_mean, weights_cov, y_k )

    return x_mean_posterior, Q_x_cov_posterior, residual
