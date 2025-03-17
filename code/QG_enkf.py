import numpy as np
from construct_GC_2d import construct_GC_2d_general
from enkf import eakf
from os.path import dirname, join as pjoin
from QG_tracer import QG_tracer
from time import time

np.random.seed(2025)

# --------------------- load data --------------------------
# train_size = 1600
train_size = 200
# test_size = 400
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_truth.npz')
data = np.load(datafname)
psi_truth = data['psi_t']
x_truth = data['x_t']
y_truth = data['y_t']
xy_truth = np.concatenate((x_truth[:,:,None], y_truth[:,:,None]), axis=2)

datafname = pjoin(data_dir, 'tracer_obs.npz')
data = np.load(datafname)
sigma_obs = data['sigma_obs']
xy_obs = data['xy_obs']

ics_psi = psi_truth[:train_size, :, :, :]
n_ics = ics_psi.shape[0]

# split training and test data set (training means tuning inflation and localzaiton)
psi_truth = psi_truth[:train_size, :, :, :]
xy_truth = xy_truth[:train_size, :, :]
xy_obs = xy_obs[:train_size, :, :]
# psi_truth = psi_truth[-test_size:, :, :, :]
# xy_truth = xy_truth[-test_size:, :, :]
# xy_obs = xy_obs[-test_size:, :, :]

# ---------------------- model parameters ---------------------
kd = 10 # Nondimensional deformation wavenumber
kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
U = 1 # Zonal shear flow
r = 9 # Nondimensional Ekman friction coefficient
nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
H = 40 # Topography parameter
dt = 1e-3 # Time step size
time_steps = 4e4 # Number of time steps
Nx = 128 # Number of grid points in each direction
dx = 2 * np.pi / Nx # Domain: [-pi, pi)^2
X, Y = np.meshgrid(np.arange(-np.pi, np.pi, dx), np.arange(-np.pi, np.pi, dx))
topo = H * (np.cos(X) + 2 * np.cos(2 * Y)) # topography
topo -= np.mean(topo)
hk = np.fft.fft2(topo)
mlocs = np.array([(ix, iy) for iy in range(Nx) for ix in range(Nx)])
mlocs = np.repeat(mlocs, 2, axis=0)
model = QG_tracer(K=Nx, kd=kd, kb=kb, U=U, r=r, nu=nu, H=H, sigma_xy=0)

# ------------------- observation parameters ------------------
L = 128 # number of tracers
obs_error_var = sigma_obs**2
obs_freq_timestep = 100
ylocs = (xy_obs[0, :, :]+ np.pi) / (2 * np.pi) * Nx
ylocs = np.repeat(ylocs, 2, axis=0)
nobs = ylocs.shape[0]
R = obs_error_var * np.eye(nobs, nobs)
nobstime = xy_obs.shape[0]

# contatenate tracer and flow variables
mlocs = np.concatenate((ylocs, mlocs), axis=0)
nmod = mlocs.shape[0]

# --------------------- DA parameters -----------------------
# analysis period
iobsbeg = 20
iobsend = -1

# eakf parameters
ensemble_size = 20
inflation_values = [1.0, 1.05] # provide multiple values if for tuning
localization_values = [8, 16] # provide multiple values if for tuning
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

# initial flow field
psi0_ens = ics_psi[np.random.randint(n_ics, size=ensemble_size), :, :, :] # shape (Nens,Nx,Nx,2)
psi0_k_ens = np.fft.fft2(psi0_ens, axes=(1,2)) # shape (Nens,Nx,Nx,2)
q0_k_ens = (psi2q_mat @ psi0_k_ens[:, :, :, :, None] + plus_hk)[:, :, :, :, 0] # shape (Nens,Nx,Nx,2)
qp0_ens = np.fft.ifft2(q0_k_ens, axes=(1,2))

# initial tracer displacements
xy0_ens = np.tile(xy_obs[0, :, :][None, :, :], (ensemble_size,1,1)) + 0.1 * np.random.randn(ensemble_size, L, 2) # shape (Nens, L, 2)
x0_ens = xy0_ens[:, :, 0]
y0_ens = xy0_ens[:, :, 1]

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
            # print(iassim)

            # EnKF step
            obsstep = iassim * obs_freq_timestep + 1
            zeakf_prior[iassim, :] = np.mean(zens, axis=0)  # prior ensemble mean
            zobs = zobs_total[iassim, :]

            # inflation RTPP
            ensmean = np.mean(zens, axis=0)
            ensp = zens - ensmean
            zens = ensmean + ensp * inflation_value

            prior_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # localization matrix        
            CMat = construct_GC_2d_general(localization_value, mlocs, ylocs, Nx)

            # serial update
            zens = eakf(ensemble_size, nobs, zens, Hk, obs_error_var, localize, CMat, zobs)
            
            # save analysis
            zeakf_analy[iassim, :] = np.mean(zens, axis=0)
            analy_spread[iassim, :] = np.std(zens, axis=0, ddof=1)

            # ensemble model integration
            if iassim < nobstime - 1:
                xy0_ens = np.reshape(zens[:, :2*L], (ensemble_size, L, 2))
                x0_ens = xy0_ens[:, :, 0]
                y0_ens = xy0_ens[:, :, 1]
                psi0_ens = np.reshape(zens[:, 2*L:], (ensemble_size, Nx, Nx, 2))
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

                # updata ylocs
                ylocs = (np.mean(xy1_ens, axis=0)+ np.pi) / (2 * np.pi) * Nx
                ylocs = np.repeat(ylocs, 2, axis=0)

        # t1 = time() - t0
        
        prior_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, 2*L:] - zeakf_prior[:, 2*L:]) ** 2, axis=1)
        analy_mse_flow[:, iinf, iloc] = np.mean((ztruth[:, 2*L:] - zeakf_analy[:, 2*L:]) ** 2, axis=1)
        prior_err_flow[iinf, iloc] = np.mean(prior_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err_flow[iinf, iloc] = np.mean(analy_mse_flow[iobsbeg - 1: iobsend, iinf, iloc])

        prior_mse_tracer[:, iinf, iloc] = np.mean((ztruth[:, :2*L] - zeakf_prior[:, :2*L]) ** 2, axis=1)
        analy_mse_tracer[:, iinf, iloc] = np.mean((ztruth[:, :2*L] - zeakf_analy[:, :2*L]) ** 2, axis=1)
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
}
np.savez('../data/qg_enkf.npz', **save)

prior_err = np.nan_to_num(prior_err_flow, nan=999999)
analy_err = np.nan_to_num(analy_err_flow, nan=999999)

# uncomment these if for tuning inflation and localization
minerr = np.min(prior_err)
inds = np.where(prior_err == minerr)
print('min prior mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))
minerr = np.min(analy_err)
inds = np.where(analy_err == minerr)
ind = inds[0][0]
print('min analy mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))

# # uncomment these if for test
# print('prior time mean mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(prior_err[0,0], inflation_values[0], localization_values[0]))
# print('analy time mean mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(analy_err[0,0], inflation_values[0], localization_values[0]))
# print('prior mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, 2*L:] - zeakf_prior[:, 2*L:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))
# print('analy mse = {0:.6e}, inflation = {1:.3f}, localizaiton = {2:d}'.format(np.mean(((ztruth[:, 2*L:] - zeakf_prior[:, 2*L:])[iobsbeg:, :]) ** 2), inflation_values[0], localization_values[0]))