import numpy as np


def eakf(ensemble_size, nobs, zens, Hk, obs_error_var, localize, CMat, zobs):
    """
    Ensemble Adjustment Kalman Filter (EAKF) tailored for Lagrangian data assimilation (passive tracer).
    
    Parameters:
        ensemble_size (int): Number of ensemble members.
        nobs (int): Number of observations.
        zens (np.ndarray): Ensemble matrix of shape (ensemble_size, nmod).
        Hk (np.ndarray): Observation operator matrix of shape (nobs, nmod).
        obs_error_var (float): Observation error variance.
        localize (int): Flag for localization (1 for applying localization, 0 otherwise).
        CMat (np.ndarray): Localization matrix of shape (nobs, nmod).
        zobs (np.ndarray): Observations of shape (nobs,).
    
    Returns:
        np.ndarray: Updated ensemble matrix.
    """
    rn = 1.0 / (ensemble_size - 1)
    
    for iobs in range(nobs):
        xmean = np.mean(zens, axis=0)
        xprime = zens - xmean
        # hxens = Hk[iobs, :] @ zens.T
        hxens = zens[:, iobs] # particularly for the tracer-flow case
        hxmean = np.mean(hxens)
        hxprime = hxens - hxmean
        hpbht = hxprime @ hxprime.T * rn
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (hxprime @ xprime) * rn

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht / (hpbht + obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        mean_inc = kfgain * (zobs[iobs] - hxmean)
        prime_inc = - (gainfact * kfgain[:, None] @ hxprime[None, :]).T
        
        zens = zens + mean_inc + prime_inc
    
    return zens