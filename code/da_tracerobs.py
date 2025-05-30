"""
Lagrangian DA for the 2-layer QG system with tracer observations
"""

import numpy as np
from Lagrangian_DA import Lagrangian_DA_OU
from LSM_QG import layer2eigen
from mode_truc import inv_truncate, truncate
from time import time
import gc

# fix the random seed
np.random.seed(2024)


# load data
data_path = '../data/qg/QG_DATA_topo40_nu1e-12_beta22_K128_dt1e-3_subs.mat'
with h5py.File(data_path, 'r') as file:
    print("Keys: %s" % file.keys())
    psi1_k_t = np.transpose(file['psi_1_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    psi2_k_t = np.transpose(file['psi_2_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    # psi1_k_t_fine = np.transpose(file['psi_1_t_fine'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    dt = file['dt'][()][0,0]
    s_rate = int(file['s_rate'][()][0,0])
    params_dataset = file['params']
    nu = params_dataset['nu'][()] [0,0]
    kd = params_dataset['kd'][()] [0,0]
    U = params_dataset['U'][()] [0,0]
    kb = params_dataset['kb'][()] [0,0]
    kappa = params_dataset['r'][()] [0,0]
    beta = kb**2
    K = int(params_dataset['N'][()] [0,0])
    H = params_dataset['H'][()] [0,0]
    topo = np.transpose(file['topo'][()], axes=(1,0))
# dt = dt * s_rate
# print('psi1_k_t.shape',psi1_k_t.shape)
psi1_k_t = psi1_k_t['real'] + 1j * psi1_k_t['imag']
psi2_k_t = psi2_k_t['real'] + 1j * psi2_k_t['imag']
# psi1_k_t_fine = psi1_k_t_fine['real'] + 1j * psi1_k_t_fine['imag']
# h_hat = np.fft.fft2(topo)

# truncate parameter
r_cut = 16
style = 'circle'

# ---------------- step 1: DA with linear stochastic flow model -------------------
# load data of LSM
eigens = np.load('../data/eigens_K128_beta22.npz')
omega1 = eigens['omega1']
omega2 = eigens['omega2']
r1 = eigens['r1']
r2 = eigens['r2']
est_params = np.load('../data/est_paras_ou_K128_beta22_tr_h40.npz')
# est_params = np.load('/grad/wang3262/Proj_1_LagrangeDA/data/est_paras_ou_K128_beta22_tr.npz')
gamma_est = est_params['gamma']
omega_est = est_params['omega']
f_est = est_params['f']
sigma_est = est_params['sigma']
# est_params = np.load('../data/est_paras_cn_ou_K128_beta22_tr.npz')
# sigma_cn = est_params['sigma']
# cov_cn = est_params['cov']
obs = np.load('../data/obs_K128_beta22_h40.npz')
xt = obs['xt']
yt = obs['yt']
sigma_xy = obs['sigma_xy']
L = 256
xt = xt[:L, :]
yt = yt[:L, :]

# initial states
psi_k_t, tau_k_t = layer2eigen(K, r_cut, r1, r2, psi1_k_t, psi2_k_t, style=style)
psi_k_t = psi_k_t[:, :, 18749:]
tau_k_t = tau_k_t[:, :, 18749:]

# initial covariance
K_ = truncate(np.ones((K,K)), r_cut, style).shape[0]
R0 = np.eye(2*K_, dtype='complex') * 1e-2

# Lagrangian DA
N_chunk = 10000
N = 100000
lsm_da = Lagrangian_DA_OU(K, r1, r2, f_est, gamma_est, omega_est, sigma_est, r_cut, style)
mu_t_lsm, R_t_lsm = lsm_da.forward(N, N_chunk, dt, s_rate=1, R0=R0, tracer=True, psi_k_t=psi_k_t, tau_k_t=tau_k_t, sigma_xy=sigma_xy, xt=xt.T, yt=yt.T)

# save data
da_pos = {
    'mu_t': mu_t_lsm,
    'R_t': R_t_lsm,
    'r_cut':r_cut,
    'style':style,
    'dt': dt,
    'L': L
}
np.savez('../data/LSMDA_pos_K128_beta22_tr_L256_h40.npz', **da_pos)

# # correlated noise
# cov = np.mean(cov_cn, axis=2)
# lsm_da = Lagrangian_DA_OU(K, r1, r2, f_est, gamma_est, omega_est, sigma_cn, r_cut, style, corr_noise=True, cov=cov)
# mu_t_cn, R_t_cn = lsm_da.forward(N, N_chunk, dt, s_rate=1, tracer=True, psi_k_t=psi_k_t, tau_k_t=tau_k_t, sigma_xy=sigma_xy, xt=xt.T, yt=yt.T)

# # save data
# da_pos = {
#     'mu_t': mu_t_cn,
#     'R_t': R_t_cn,
#     'r_cut':r_cut,
#     'style':style,
#     'dt': dt
# }
# np.savez('../data/LSMDA_cn_pos_K128_beta22_tr_L256.npz', **da_pos)


# # ---------------- step 2: DA with CG nonlinear stochastic flow model -------------------
# da_pos = np.load('../data/LSMDA_pos_K128_beta22_tr_L256.npz')
# mu_t_lsm = da_pos['mu_t']
# R_t_lsm = da_pos['R_t']

# # N_chunk = 5000
# # cg_da = Lagrangian_DA_CG(K, kd, beta, kappa, nu, U, h_hat, r_cut, style)
# # Sigma1, Sigma2 = cg_da.calibrate_sigma(N_chunk, dt*s_rate, psi1_k_t, psi2_k_t)

# # # save data
# # data = {
# #     'Sigma1': Sigma1,
# #     'Sigma2': Sigma2
# # }
# # np.savez('../data/Sigma_cali_CGDA_K128_beta22_tr.npz', **data)

# # load data
# data = np.load('../data/Sigma_cali_CGDA_K128_beta22_tr.npz')
# Sigma1 = data['Sigma1']
# Sigma2 = data['Sigma2']

# # Lagrangian DA
# N = 100000 
# N_chunk = 5000
# N_s = 16 # number of sample trajectories

# # minimize RAM usage
# psi2_k_t = psi2_k_t[:,:,-N//s_rate]
# mu_t_lsm = mu_t_lsm[:,-N:]
# R_t_lsm = R_t_lsm[:,-N:]
# del da_pos
# gc.collect()  # Explicitly trigger garbage collection

# cg_da = Lagrangian_DA_CG(K, kd, beta, kappa, nu, U, h_hat, r_cut, style)
# mu_t, R_t = cg_da.forward(N, N_chunk, dt, N_s, Sigma1, Sigma2, s_rate, psi2_k_t0=psi2_k_t, mu_eigen_t=mu_t_lsm , R_eigen_t=R_t_lsm , sigma=sigma_est, f=f_est, gamma=gamma_est, omega=omega_est, r1=r1, r2=r2)

# # mu_t_mean = np.mean(mu_t, axis=0)
# # R_t_mean = np.mean(R_t, axis=0) + np.var(mu_t, axis=0)

# # save data
# da_pos = {
#     'mu_t': mu_t,
#     'R_t': R_t,
#     # 'mu_t_mean': mu_t_mean,
#     # 'R_t_mean': R_t_mean,
#     'r_cut':r_cut,
#     'style':style,
#     'dt': dt
# }
# np.savez('../data/CGDA_ens16_pos_K128_beta22_tr_L256.npz', **da_pos)

# # # constant covariance
# # # load equilibrium R0
# # R0 = np.load('../data/R0_cgda_e16_b22_L256.npy')
# # R0 = np.diag(R0)

# # cg_da = Lagrangian_DA_CG(K, kd, beta, kappa, nu, U, h_hat, r_cut, style)
# # mu_t, R_t = cg_da.forward(N, N_chunk, dt, N_s, Sigma1, Sigma2, s_rate, R0, forward_R=False, psi2_k_t0=psi2_k_t, mu_eigen_t=mu_t_lsm , R_eigen_t=R_t_lsm , sigma=sigma_est, f=f_est, gamma=gamma_est, omega=omega_est, r1=r1, r2=r2)

# # # mu_t_mean = np.mean(mu_t, axis=0)
# # # R_t_mean = np.mean(R_t, axis=0) + np.var(mu_t, axis=0)

# # # save data
# # da_pos = {
# #     'mu_t': mu_t,
# #     'R_t': R_t,
# #     # 'mu_t_mean': mu_t_mean,
# #     # 'R_t_mean': R_t_mean,
# #     'r_cut':r_cut,
# #     'style':style,
# #     'dt': dt
# # }
# # np.savez('../data/CGDA_ens16_pos_K128_beta22_tr_L256_constR.npz', **da_pos)