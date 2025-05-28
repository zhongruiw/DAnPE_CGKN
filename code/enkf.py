import numpy as np
from numpy.linalg import inv, norm
from numba import jit, prange

def construct_GC_2d_general(cut, mlocs, ylocs, Nx=None):
    """
    Construct the Gaspari and Cohn localization matrix for a 2D field.

    Parameters:
        cut (float): Localization cutoff distance.
        mlocs (array of shape (nstates,2)): 2D coordinates of the model states [(x1, y1), (x2, y2), ...].
        ylocs (array of shape (nobs,2)): 2D coordinates of the observations [[x1, y1], [x2, y2], ...].
        Nx (int, optional): Number of grid points in each direction.

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(mlocs)).
    """
    ylocs = ylocs[:, np.newaxis, :]  # Shape (nobs, 1, 2)
    mlocs = mlocs[np.newaxis, :, :]  # Shape (1, nstates, 2)

    # Compute distances
    dist = np.linalg.norm((mlocs - ylocs + Nx // 2) % Nx - Nx // 2, axis=2)

    # Normalize distances
    r = dist / (0.5 * cut)

    # Compute localization function
    V = np.zeros_like(dist)

    mask2 = (0.5 * cut <= dist) & (dist < cut)
    mask3 = (dist < 0.5 * cut)

    V[mask2] = (
        r[mask2]**5 / 12.0 - r[mask2]**4 / 2.0 + r[mask2]**3 * 5.0 / 8.0
        + r[mask2]**2 * 5.0 / 3.0 - 5.0 * r[mask2] + 4.0 - 2.0 / (3.0 * r[mask2])
    )
    
    V[mask3] = (
        -r[mask3]**5 * 0.25 + r[mask3]**4 / 2.0 + r[mask3]**3 * 5.0 / 8.0 
        - r[mask3]**2 * 5.0 / 3.0 + 1.0
    )

    return V

def error_bench(cov, H, var_obs):
    nobs = H.shape[0]
    return np.trace(cov - cov @ H.T @ inv(var_obs * np.eye(nobs) + H @ cov @ H.T) @ H @ cov) 

def thresholds(err_bench, H, nobs, var_obs, N_ens):
    M1 = np.sqrt(norm(H)**2 * err_bench + 2 * nobs * var_obs)
    M2 = N_ens / (N_ens - 2) * err_bench

    return M1, M2

def adapt_add_inflation(c, obs_inc_ens, hp, M1, M2, nobs):
    '''
    obs_inc_ens: shape (N_ens, nobs)
    hp: shape (nobs, nmod)
    '''
    theta = np.sqrt(np.mean(np.sum(obs_inc_ens**2, axis=1)))
    xi = norm(hp[:, nobs:], ord=2)
    
    if theta > M1 or xi > M2:
        inflation = c * theta * (1 + xi)
    else:
        inflation = 0

    return inflation, theta, xi


@jit(nopython=True)
def periodic_mean(x, size):
    '''
    compute the periodic mean for a single x of shape (size,) on domain [0, 2pi)
    '''
    # if np.max(x) - np.min(x) <= np.pi:
    if np.ptp(x) <= np.pi:
        x_mean = np.mean(x)
    else:
        # calculate the periodic mean of two samples
        x_mean = np.mod(np.arctan2((np.sin(x[0]) + np.sin(x[1])) / 2, (np.cos(x[0]) + np.cos(x[1])) / 2), 2*np.pi)

        # successively calculate the periodic mean for the rest samples
        for k in range(3, size+1):
            inc = np.mod(x[k-1] - x_mean + np.pi, 2*np.pi) - np.pi # increment with periodicity considered
            x_mean = x_mean + inc / k
        x_mean = np.mod(x_mean, 2*np.pi)

    return x_mean

@jit(nopython=True, parallel=True, fastmath=True)
def periodic_means(xs, size, L):
    '''
    compute the periodic means of multiple x of shape (size, L)
    '''
    means = np.zeros(L)
    for l in range(L):
        means[l] = periodic_mean(xs[:, l], size)

    return means
        

def eakf(ensemble_size, nobs, xens, Hk, obs_error_var, localize, CMat, obs, inflate=0, c=0, M1=0, M2=0):
    """
    Ensemble Adjustment Kalman Filter (EAKF) tailored for Lagrangian data assimilation.
    Lagrangian observations are passive tracer locations on [0, 2*pi)^2 domain.

    Parameters:
        ensemble_size (int): Number of ensemble members.
        nobs (int): Number of observations.
        xens (np.ndarray): Ensemble matrix of shape (ensemble_size, nmod).
        Hk (np.ndarray): Observation operator matrix of shape (nobs, nmod).
        obs_error_var (float): Observation error variance.
        localize (int): Flag for localization (1 for applying localization, 0 otherwise).
        CMat (np.ndarray): Localization matrix of shape (nobs, nmod).
        obs (np.ndarray): Observations of shape (nobs,).
    
    Returns:
        np.ndarray: Updated ensemble matrix.
    """
    rn = 1.0 / (ensemble_size - 1)
    xmean = np.zeros(xens.shape[1])
    xprime = np.zeros((ensemble_size, xens.shape[1]))
    # inflation_record = np.zeros((nobs,3))
    
    for iobs in range(nobs):
        xmean[:nobs] = periodic_means(xens[:, :nobs], ensemble_size, nobs) # tracer mean
        xmean[nobs:] = np.mean(xens[:, nobs:], axis=0) # flow mean
        xprime = xens - xmean
        xprime[:, :nobs] = np.mod(xprime[:, :nobs] + np.pi, 2*np.pi) - np.pi # tracer perturbation
        # hxens = Hk[iobs, :] @ xens.T
        # hxmean = np.mean(hxens)
        # hxprime = hxens - hxmean
        hxens = xens[:, iobs] # particularly for the tracer-flow case
        hxmean = xmean[iobs] # particularly for the tracer-flow case
        hxprime = xprime[:, iobs] # particularly for the tracer-flow case
        hpbht = hxprime @ hxprime.T * rn
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (hxprime @ xprime) * rn        
        obs_inc = np.mod(obs[iobs] - hxmean + np.pi, 2*np.pi) - np.pi

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht / (hpbht + obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        prime_inc = - (gainfact * kfgain[:, None] @ hxprime[None, :]).T

        if inflate == 1:
            obs_inc_ens = np.mod(obs[iobs] - hxens + np.pi, 2*np.pi) - np.pi
            inflation, theta, xi = adapt_add_inflation(c, obs_inc_ens[:, None], pbht[None, :], M1, M2, nobs)
            # inflation_record[iobs, 0] = inflation
            # inflation_record[iobs, 1] = theta
            # inflation_record[iobs, 2] = xi
            if inflation != 0:
                hpbht_inf = hpbht + inflation
                pbht_inf = pbht + inflation * Hk[iobs, :]
                if localize == 1:
                    Cvect = CMat[iobs, :]
                    kfgain_inf = Cvect * (pbht_inf / (hpbht_inf + obs_error_var))
                else:
                    kfgain_inf = pbht_inf / (hpbht_inf + obs_error_var)

                mean_inc = kfgain_inf * obs_inc
            else:
                mean_inc = kfgain * obs_inc
        else:
            mean_inc = kfgain * obs_inc
        
        xens = xens + mean_inc + prime_inc
        xens[:, :nobs] = np.mod(xens[:, :nobs], 2*np.pi) # periodic condition for tracer
    
    return xens#, inflation_record