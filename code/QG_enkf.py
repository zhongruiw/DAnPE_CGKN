import numpy as np
from construct_GC_2d import construct_GC_2d_general
from enkf import eakf, periodic_means
from os.path import dirname, join as pjoin
from QG_tracer import QG_tracer
from time import time

np.random.seed(2025)

# --------------------- load data --------------------------
train_size = 8000
test_size = 2000
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_truth_new.npz')
data = np.load(datafname)
psi_truth_full = data['psi_t']
xy_truth_full = np.concatenate((data['x_t'][:,:,None], data['y_t'][:,:,None]), axis=2)

data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_data_new.npz')
data = np.load(datafname)
xy_obs_full = data['xy_obs']
sigma_obs = data['sigma_obs']
dt_ob = data['dt_ob']

# split training and test data set (training means tuning inflation and localzaiton)
data_size = test_size
# psi_truth = psi_truth_full[:data_size, :, :, :]
# xy_truth = xy_truth_full[:data_size, :, :]
# xy_obs = xy_obs_full[:data_size, :, :]
psi_truth = psi_truth_full[-data_size:, :, :, :]
xy_truth = xy_truth_full[-data_size:, :, :]
xy_obs = xy_obs_full[-data_size:, :, :]

# ---------------------- model parameters ---------------------
kd = 10 # Nondimensional deformation wavenumber
kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
U = 1 # Zonal shear flow
r = 9 # Nondimensional Ekman friction coefficient
nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
H = 40 # Topography parameter
dt = 1e-3 # Time step size
Nx = 128 # Number of grid points in each direction
dx = 2 * np.pi / Nx # Domain: [0, 2pi)^2
X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
topo = H * (np.cos(X) + 2 * np.cos(2 * Y)) # topography
topo -= np.mean(topo)
hk = np.fft.fft2(topo)
mlocs = np.array([(ix, iy) for iy in range(Nx) for ix in range(Nx)])
mlocs = np.repeat(mlocs, 2, axis=0)
model = QG_tracer(K=Nx, kd=kd, kb=kb, U=U, r=r, nu=nu, H=H, sigma_xy=0)

# ------------------- observation parameters ------------------
L = 128 # number of tracers
obs_error_var = sigma_obs**2
obs_freq_timestep = int(dt_ob / dt)
ylocs = xy_obs[0, :, :] / (2 * np.pi) * Nx
ylocs = np.repeat(ylocs, 2, axis=0)
nobs = ylocs.shape[0]
# R = obs_error_var * np.eye(nobs, nobs)
nobstime = xy_obs.shape[0]

# contatenate tracer and flow variables
mlocs = np.concatenate((ylocs, mlocs), axis=0)
nmod = mlocs.shape[0]

# --------------------- DA parameters -----------------------
# analysis period
iobsbeg = 20
iobsend = -1

# eakf parameters
ensemble_size = 100
inflation_values = [1.0] # provide multiple values if for tuning
localization_values = [16] # provide multiple values if for tuning
ninf = len(inflation_values)
nloc = len(localization_values)
localize = 1

# ---------------------- initialization -----------------------
# prepare matrix
Kx = np.fft.fftfreq(Nx) * Nx
Ky = np.fft.fftfreq(Nx) * Nx
KX, KY = np.meshgrid(Kx, Ky)
K_square = KX**2 + KY**2
psi2q_mat = (-np.eye(2) * K_square[:, :, None, None]) + (kd**2/2 * np.ones((2,2)) - np.eye(2) * kd**2)
psi2q_mat = psi2q_mat.astype(np.complex128)
plus_hk = np.zeros(((Nx,Nx,2,1)), dtype=complex)
plus_hk[:,:,1,0] = hk
Hk = np.zeros((nobs, nmod)) # observation forward operator H
Hk[:, :nobs] = np.eye(nobs, nobs)

# ics_psi = psi_truth_full[:train_size, :, :, :]
# n_ics = ics_psi.shape[0]
spinup_flow = 25
spinup_tracer = 10
spinup_total = spinup_flow + spinup_tracer

# initial flow field
# psi0_ens = ics_psi[np.random.randint(n_ics, size=ensemble_size), :, :, :] # shape (Nens,Nx,Nx,2)
psi0_ens =  np.tile(psi_truth_full[-(spinup_total+data_size),:, :, :][None, :, :, :], (ensemble_size,1,1,1)) + 0.1 * np.random.randn(ensemble_size,Nx,Nx,2)
psi0_k_ens = np.fft.fft2(psi0_ens, axes=(1,2)) # shape (Nens,Nx,Nx,2)
q0_k_ens = (psi2q_mat @ psi0_k_ens[:, :, :, :, None] + plus_hk)[:, :, :, :, 0] # shape (Nens,Nx,Nx,2)
qp0_ens = np.real(np.fft.ifft2(q0_k_ens, axes=(1,2)))

_, _, _, qp0_ens = model.forward_ens(ens=ensemble_size, Nt=spinup_flow*obs_freq_timestep, dt=dt, qp_ens=qp0_ens, L=L, x0=np.zeros((ensemble_size,L)), y0=np.zeros((ensemble_size,L)))
qp0_ens = qp0_ens[:, -1, :, :, :]

# initial tracer displacements
# xy0_ens = np.tile(xy_obs[0, :, :][None, :, :], (ensemble_size,1,1)) + 0.1 * np.random.randn(ensemble_size, L, 2) # shape (Nens, L, 2)
xy0_ens = np.tile(xy_truth_full[-(spinup_tracer+data_size), :, :][None, :, :], (ensemble_size,1,1))
x0_ens = xy0_ens[:, :, 0]
y0_ens = xy0_ens[:, :, 1]

psi0_k_ens, x0_ens, y0_ens, _ = model.forward_ens(ens=ensemble_size, Nt=spinup_tracer*obs_freq_timestep, dt=dt, qp_ens=qp0_ens, L=L, x0=x0_ens, y0=y0_ens)
psi0_k_ens = psi0_k_ens[:, -1, :, :, :]
x0_ens = x0_ens[:, :, -1]
y0_ens = y0_ens[:, :, -1]
xy0_ens = np.concatenate((x0_ens[:, :, None], y0_ens[:, :, None]), axis=2)
psi0_ens = np.real(np.fft.ifft2(psi0_k_ens, axes=(1,2)))

# initial augmented variable
psi0_ens_flat = np.reshape(psi0_ens, (ensemble_size, -1)) # shape (Nens, Nx*Nx*2)
xy0_ens_flat = np.reshape(xy0_ens, (ensemble_size, -1)) # shape (Nens, L*2)
z0_ens = np.concatenate((xy0_ens_flat, psi0_ens_flat), axis=1) # shape (Nens, L*2+Nx*Nx*2)
zobs_total = np.reshape(xy_obs, (nobstime, -1))
ztruth = np.concatenate((np.reshape(xy_truth, (nobstime, -1)), np.reshape(psi_truth, (nobstime, -1))), axis=1)

prior_mse_flow = np.zeros((nobstime,ninf,nloc))
analy_mse_flow = np.zeros((nobstime,ninf,nloc))
prior_err_flow = np.zeros((ninf,nloc))
analy_err_flow = np.zeros((ninf,nloc))
prior_mse_tracer = np.zeros((nobstime,ninf,nloc))
analy_mse_tracer = np.zeros((nobstime,ninf,nloc))
prior_err_tracer = np.zeros((ninf,nloc))
analy_err_tracer = np.zeros((ninf,nloc))
# PHt = np.zeros((nobstime, nmod, nobs))
# PHt_local = np.zeros((nobstime, nmod, nobs))

# ---------------------- assimilation -----------------------
for iinf in range(ninf):
    inflation_value = inflation_values[iinf]
    print('inflation:',inflation_value)
    
    for iloc in range(nloc):
        localization_value = localization_values[iloc]
        print('localization:',localization_value)

        zens = z0_ens
        zeakf_prior = np.zeros((nobstime, nmod))
        zeakf_analy = np.empty((nobstime, nmod))
        prior_spread = np.empty((nobstime, nmod))
        analy_spread = np.empty((nobstime, nmod))

        # t0 = time()
        for iassim in range(0, nobstime):
            print(iassim)

            # EnKF step
            obsstep = iassim * obs_freq_timestep + 1
            zeakf_prior[iassim, :nobs] = periodic_means(zens[:, :nobs], ensemble_size, nobs)
            zeakf_prior[iassim, nobs:] = np.mean(zens[:, nobs:], axis=0)  # prior ensemble mean
            
            zobs = zobs_total[iassim, :]

            # # inflation RTPP
            # ensmean = np.mean(zens, axis=0)
            # ensp = zens - ensmean
            # zens = ensmean + ensp * inflation_value

            # # adjustments needed to ensure periodicity of tracer positions if inflation

            prior_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # localization matrix        
            CMat = construct_GC_2d_general(localization_value, mlocs, ylocs, Nx)
            np.fill_diagonal(CMat[:nobs, :nobs], 1) # the tracer observation with the same ID has localization value of 1

            # # check Kalman gain
            # HXprime = np.mod(zens[:, :nobs] - periodic_means(zens[:, :nobs], ensemble_size, nobs) + np.pi, 2*np.pi) - np.pi # tracer mean
            # Xprime = zens - np.mean(zens, axis=0)
            # Xprime[:, :nobs] = HXprime
            # PHt[iassim, :, :] = Xprime.T @ HXprime / (ensemble_size - 1)
            # PHt_local[iassim, :, :] = PHt[iassim, :, :] * CMat.T
            
            # serial update
            zens = eakf(ensemble_size, nobs, zens, Hk, obs_error_var, localize, CMat, zobs)
            
            # save analysis
            zeakf_analy[iassim, :nobs] = periodic_means(zens[:, :nobs], ensemble_size, nobs)
            zeakf_analy[iassim, nobs:] = np.mean(zens[:, nobs:], axis=0)
            analy_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # ensemble model integration
            if iassim < nobstime - 1:
                xy0_ens = np.reshape(zens[:, :nobs], (ensemble_size, L, 2))
                x0_ens = xy0_ens[:, :, 0]
                y0_ens = xy0_ens[:, :, 1]
                psi0_ens = np.reshape(zens[:, nobs:], (ensemble_size, Nx, Nx, 2))
                psi0_k_ens = np.fft.fft2(psi0_ens, axes=(1,2)) # shape (Nens,Nx,Nx,2)
                q0_k_ens = (psi2q_mat @ psi0_k_ens[:, :, :, :, None] + plus_hk)[:, :, :, :, 0] # shape (Nens,Nx,Nx,2)
                qp0_ens = np.real(np.fft.ifft2(q0_k_ens, axes=(1,2)))

                psi1_k_ens, x1_ens, y1_ens, _ = model.forward_ens(ens=ensemble_size, Nt=obs_freq_timestep, dt=dt, qp_ens=qp0_ens, L=L, x0=x0_ens, y0=y0_ens)
                
                psi1_k_ens = psi1_k_ens[:, -1, :, :, :]
                x1_ens = x1_ens[:, :, -1]
                y1_ens = y1_ens[:, :, -1]
                xy1_ens = np.concatenate((x1_ens[:, :, None], y1_ens[:, :, None]), axis=2)
                psi1_ens = np.real(np.fft.ifft2(psi1_k_ens, axes=(1,2)))

                zens = np.concatenate((np.reshape(xy1_ens, (ensemble_size, -1)), np.reshape(psi1_ens, (ensemble_size, -1))), axis=1)

                # updata tracer locations
                ylocs = xy_obs[iassim, :, :] / (2 * np.pi) * Nx
                ylocs = np.repeat(ylocs, 2, axis=0)
                mlocs_tracer = np.mean(xy1_ens, axis=0) / (2 * np.pi) * Nx
                mlocs_tracer = np.repeat(mlocs_tracer, 2, axis=0)
                mlocs[:2*L, :] = mlocs_tracer

        # t1 = time() - t0
        
        prior_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, nobs:] - zeakf_prior[:, nobs:]) ** 2, axis=1)
        analy_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, nobs:] - zeakf_analy[:, nobs:]) ** 2, axis=1)
        prior_err_flow[iinf, iloc] = np.mean(prior_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_flow[iinf, iloc] = np.mean(analy_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])

        prior_mse_tracer[:, iinf, iloc] = np.mean((np.mod(ztruth[:, :nobs] - zeakf_prior[:, :nobs] + np.pi, 2*np.pi) - np.pi) ** 2, axis=1)
        analy_mse_tracer[:, iinf, iloc] = np.mean((np.mod(ztruth[:, :nobs] - zeakf_analy[:, :nobs] + np.pi, 2*np.pi) - np.pi) ** 2, axis=1)
        prior_err_tracer[iinf, iloc] = np.mean(prior_mse_tracer[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_tracer[iinf, iloc] = np.mean(analy_mse_tracer[iobsbeg - 1: iobsend, iinf, iloc])

save = {
    'mean_analy': zeakf_analy,
    'mean_prior': zeakf_prior,
    'spread_analy': analy_spread,
    'spread_prior': prior_spread,
    'mse_prior_flow': prior_mse_flow,
    'mse_analy_flow': analy_mse_flow,
    'mse_prior_tracer': prior_mse_tracer,
    'mse_analy_tracer': analy_mse_tracer,
    # 'PHt': PHt,
    # 'PHt_local': PHt_local
}
np.savez('../data/qg_enkf_new.npz', **save)

prior_err = np.nan_to_num(prior_err_flow, nan=999999)
analy_err = np.nan_to_num(analy_err_flow, nan=999999)

# # uncomment these if for tuning inflation and localization
# minerr = np.min(prior_err)
# inds = np.where(prior_err == minerr)
# print('min prior mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))
# minerr = np.min(analy_err)
# inds = np.where(analy_err == minerr)
# ind = inds[0][0]
# print('min analy mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))

# uncomment these if for test
print('prior time mean mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(prior_err[0,0], inflation_values[0], localization_values[0]))
print('analy time mean mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(analy_err[0,0], inflation_values[0], localization_values[0]))
print('prior mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, nobs:] - zeakf_prior[:, nobs:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))
print('analy mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, nobs:] - zeakf_analy[:, nobs:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))