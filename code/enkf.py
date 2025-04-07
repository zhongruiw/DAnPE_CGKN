import numpy as np
from numba import jit, prange


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
        

def eakf(ensemble_size, nobs, xens, Hk, obs_error_var, localize, CMat, obs):
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

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht / (hpbht + obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        obs_inc = np.mod(obs[iobs] - hxmean + np.pi, 2*np.pi) - np.pi
        mean_inc = kfgain * obs_inc
        prime_inc = - (gainfact * kfgain[:, None] @ hxprime[None, :]).T
        
        xens = xens + mean_inc + prime_inc
        xens[:, :nobs] = np.mod(xens[:, :nobs], 2*np.pi) # periodic condition for tracer
    
    return xens