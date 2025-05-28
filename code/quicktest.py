import numpy as np
from os.path import dirname, join as pjoin
from time import time

"""
State estimation and parameter estimation for the 2-layer QG system with only surface observations (hidden flow states and unknown topography)
"""
from Lagrangian_tracer import Lagrange_tracer_model
from conj_symm_tools import verify_conjugate_symmetry, find_non_conjugate_pairs, avg_conj_symm, map_conj_symm
from Lagrangian_DA import Lagrangian_DA_OU, Lagrangian_DA_CG, mu2psi, mu2layer, R2layer, relative_entropy_psi_k
from ene_spectrum import ene_spectrum, adjust_ik, trunc2full
from LSM_QG import solve_eigen, calibrate_OU, run_OU, eigen2layer, layer2eigen, growth_rate
from mode_truc import inv_truncate, truncate
from plot import ifftncheck, psi2q, plot_contour_fields, plot_psi_k_seriespdf, plot_layer_seriespdf, plot_psi1_k_seriespdf, plot_rmses, loop_ifft2_var, ifft2_var, plot_mog, plot_mog_k, scatterplot, calculate_skewness_kurtosis
import h5py
from time import time

# fix the random seed
np.random.seed(2024)

save = np.load('../data/quicktest_discrete.npz')
sigma_xy = save['sigma_xy'].item()
psi_k_t = save['psi_k_t']
x_t = save['x_t']
y_t = save['y_t']

sigma_obs = 0.2
x_obs = x_t + sigma_obs * np.random.randn(x_t.shape[0], x_t.shape[1])
y_obs = y_t + sigma_obs * np.random.randn(y_t.shape[0], y_t.shape[1])

eigens = np.load('../data/eigens.npz')
r1 = eigens['r1']
r2 = eigens['r2']

est_params = np.load('../data/est_paras_ou.npz')

gamma_est = est_params['gamma']
omega_est = est_params['omega']
f_est = est_params['f']
sigma_est = est_params['sigma']

psi1_k_t = np.transpose(psi_k_t[:, :, :, 0], axes=(1,2,0))
psi2_k_t = np.transpose(psi_k_t[:, :, :, 1], axes=(1,2,0))

dt = 1e-3
K = 128
# truncate parameter
r_cut = 16
style = 'circle'

# --------------------- load data --------------------------
train_size = 8000
test_size = 2000
data_dir = '../data/'
datafname = pjoin(data_dir, 'qg_truth_new.npz')
data = np.load(datafname)
psi_truth_full = data['psi_t']

# ------------------- observation parameters ------------------
L = 128 # number of tracers
K = 128
ics_psi = psi_truth_full[:train_size, :, :, :]
n_ics = ics_psi.shape[0]
psi0_ens = ics_psi[np.random.randint(n_ics, size=1), :, :, :] # shape (Nens,Nx,Nx,2)

ics_psi_k = np.fft.fft2(ics_psi, axes=(1,2))
psi1_k_ics = np.transpose(ics_psi_k[:, :, :, 0], axes=(1,2,0))
psi2_k_ics = np.transpose(ics_psi_k[:, :, :, 1], axes=(1,2,0))
psi_k_ics, tau_k_ics = layer2eigen(K, r_cut, r1, r2, psi1_k_ics, psi2_k_ics, style=style)

mu_ics = np.concatenate((truncate(psi_k_ics,r_cut, style=style), truncate(tau_k_ics,r_cut, style=style)))
R0_diag = np.var(mu_ics, axis=1)

psi0 = psi0_ens[0, :, :, :]
psi0_k = np.fft.fft2(psi0, axes=(0,1))
psi1_k_t0 = psi0_k[:, :, 0][:, :, None]
psi2_k_t0 = psi0_k[:, :, 1][:, :, None]
psi_k_t0, tau_k_t0 = layer2eigen(K, r_cut, r1, r2, psi1_k_t0, psi2_k_t0, style=style)

# initial covariance
K_ = truncate(np.ones((K,K)), r_cut, style).shape[0]
R0 = (np.eye(2*K_) * R0_diag) * 1e-2 + np.eye(2*K_, dtype='complex') *1e-2

# Lagrangian DA
N_chunk = 5000
N = 5000
lsm_da = Lagrangian_DA_OU(K, r1, r2, f_est, gamma_est, omega_est, sigma_est, r_cut, style)
mu_t_lsm, R_t_lsm = lsm_da.forward(N, N_chunk, dt, s_rate=1, R0=R0, tracer=True, psi_k_t=psi_k_t0, tau_k_t=tau_k_t0, sigma_xy=2, xt=x_obs, yt=y_obs)

save = {
    'mu_t_lsm': mu_t_lsm,
    'R_t_lsm': R_t_lsm,
}
np.savez('../data/quicktest_sigmaxy02.npz', **save)