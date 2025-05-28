"""
Lagrangian DA for the 2-layer QG system with tracer observations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm
from Lagrangian_tracer import Lagrange_tracer_model
from conj_symm_tools import verify_conjugate_symmetry, find_non_conjugate_pairs, avg_conj_symm, map_conj_symm
from Lagrangian_DA import Lagrangian_DA_OU, Lagrangian_DA_CG, mu2psi, mu2layer, back_sampling
from ene_spectrum import ene_spectrum, adjust_ik, trunc2full
from LSM_QG import solve_eigen, calibrate_OU, run_OU, eigen2layer, layer2eigen, growth_rate
from mode_truc import inv_truncate, truncate
from plot import ifftnroll, psi2q, plot_contour_fields, plot_psi_k_seriespdf, plot_layer_seriespdf
from statsmodels.tsa.stattools import acf, ccf
from scipy.optimize import curve_fit
from scipy.io import loadmat
import h5py
from scipy import sparse
from time import time
import gc

from numba import jit
from numba.typed import Dict
from numba.core import types
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@jit(nopython=True)
def params_est_loop(N, dt, k_left, KX, KY, psi1_k_t, psi2_k_t, k_index_map, index_dic_k_map, kd=10, beta=22, kappa=9, nu=1e-12, U=1):
    # Precompute constants
    K_squared = KX**2 + KY**2
    K_squared_kd2 = K_squared + kd**2 / 2
    K_squared4 = K_squared**4
    dX = 1j * KX
    
    A = np.zeros((N*k_left, k_left), dtype=np.complex128)
    B = np.zeros(N*k_left, dtype=np.complex128)
    for i in range(N):
        psi1_k = psi1_k_t[:, i+1]
        psi2_k = psi2_k_t[:, i+1]
        q2_k1 = -K_squared_kd2 * psi2_k + kd**2/2 * psi1_k
        psi1_k = psi1_k_t[:, i]
        psi2_k = psi2_k_t[:, i]
        q2_k0 = -K_squared_kd2 * psi2_k + kd**2/2 * psi1_k
        dqdt = (q2_k1 - q2_k0) / dt

        # linear part
        linear_A = -dX * U + nu * K_squared4
        linear_B = -dX * ((beta + U*K_squared - kd**2/2*U) * psi2_k - kd**2/2*U * psi1_k) + kappa * K_squared * psi2_k - nu * K_squared4 * q2_k0

        # nonlinear part
        nonlinear_A = np.zeros((k_left, k_left), dtype=np.complex128)
        nonlinear_B = np.zeros(k_left, dtype=np.complex128)
        for ik_, (k, ik) in enumerate(k_index_map.items()):
            kx, ky = k
            for im_, (m, im) in enumerate(k_index_map.items()):
                mx, my = m
                m_sq = mx**2 + my**2
                psi1_m = psi1_k[im_]
                psi2_m = psi2_k[im_]
                n = (kx-mx, ky-my)
                if n in k_index_map:
                    in_ = index_dic_k_map[n]
                    psi2_n = psi2_k[in_]
                    det_mn = np.linalg.det(np.array([m, n]))
                    nonlinear_A[ik_, im_] = det_mn * psi2_n
                    nonlinear_B[ik_] -= det_mn * psi2_n * (-(m_sq + kd**2/2) * psi2_m + kd**2/2 * psi1_m)

        # assemble
        A[i*k_left:(i+1)*k_left, :] = np.diag(linear_A) + nonlinear_A
        B[i*k_left:(i+1)*k_left] = linear_B + nonlinear_B - dqdt

    return A, B

def params_est(K, dt, psi1_k_t, psi2_k_t, kd=10, beta=22, kappa=9, nu=1e-12, U=1, r_cut=16, style='circle'):
    '''
    parameter estimation via linear regression to solve the least square problem: 
    Ah=B
    A: training data of shape (Nx|Ktr|,|Ktr|);
    h: parameters to be estimated;
    B: RHS of shape (|Ktr|x1)
    where 
    N is the number of time steps, 
    |Ktr| is the cardinality of the truncated Fourier modes set, 

    Inputs:
    K: number of Fourier modes in one dimension
    dt: time step size
    psi1_k_t: np.array of shape (K, K, N+1)
    psi2_k_t: np.array of shape (K, K, N+1)
    '''
    # Fourier wavenumber mesh
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    KX, KY = np.meshgrid(Kx, Ky)

    # Create an empty Numba-typed dictionary
    k_index_map = Dict.empty(
        key_type=types.Tuple([types.float64, types.float64]),  
        value_type=types.Tuple([types.int64, types.int64])   
    )
    # Populate the Numba-typed dictionary using a loop
    for ix in range(K):
        for iy in range(K):
            kx = KX[iy, ix]
            ky = KY[iy, ix]
            if (kx**2 + ky**2) <= r_cut**2:
                k_index_map[(kx, ky)] = (ix, iy)

    # Define a Numba-typed dictionary of the index of flattened K_
    index_dic_k_map = Dict.empty(
        key_type=types.Tuple([types.float64, types.float64]), 
        value_type=types.int64
    )
    # Populate the Numba dictionary
    for idx, key in enumerate(k_index_map):
        index_dic_k_map[key] = idx

    # mode truncation
    psi1_k_t = truncate(psi1_k_t, r_cut, style)
    psi2_k_t = truncate(psi2_k_t, r_cut, style)
    k_left, N = psi1_k_t.shape
    N -= 1
    KX = truncate(KX, r_cut, style)
    KY = truncate(KY, r_cut, style)

    print('begin compute A, B matrices...')

    # compute A, B matrices
    A, B = params_est_loop(N, dt, k_left, KX, KY, psi1_k_t, psi2_k_t, k_index_map, index_dic_k_map, kd, beta, kappa, nu, U)

    print('successfully compute A, B matrices...')

    # seperate real and imaginary parts
    A_new = np.zeros((2*N*k_left, 2*k_left))
    B_new = np.zeros(2*N*k_left)
    A_new[:N*k_left, :k_left] = A.real
    A_new[:N*k_left, k_left:] = -A.imag
    A_new[N*k_left:, :k_left] = A.imag
    A_new[N*k_left:, k_left:] = A.real
    B_new[:N*k_left] = B.real
    B_new[N*k_left:] = B.imag

    print('successfully seperate real and imaginary parts...')
    # linear regression
    # reg = LinearRegression(fit_intercept=False).fit(A_new, B_new)
    # hk_ = reg.coef_

    # Create a pipeline to standardize the data and fit the SGDRegressor
    reg = make_pipeline(
        StandardScaler(),  # Scale features to improve optimization
        SGDRegressor(max_iter=1000, tol=1e-3, fit_intercept=False)  # Disable intercept
    )
    # Fit the SGDRegressor on the data
    reg.fit(A_new, B_new)
    # Access the scaler and regressor from the pipeline
    scaler = reg.named_steps['standardscaler']
    sgd = reg.named_steps['sgdregressor']
    # Adjust coefficients for the original data
    hk_ = sgd.coef_ / scaler.scale_

    print('successfully regression...')

    hk = hk_[:k_left] + 1j * hk_[k_left:]
    hk = inv_truncate(hk[:, None], r_cut, K, style)

    print('finished.')
    return hk


# fix the random seed
np.random.seed(2025)

dt = 1e-3
N = 100000
K = 128
s_rate = 16
eigens = np.load('../data/eigens_K128_beta22.npz')
r1 = eigens['r1']
r2 = eigens['r2']
# truncate parameter
r_cut = 16
style = 'circle'

# da_pos = np.load('../data/LSMDA_pos_K128_beta22_tr_L256_iter2.npz')
# mu_t_lsm = da_pos['mu_t']

# # reshape flattened variables to two modes matrices
# psi_k_pos_lsm, tau_k_pos_lsm = mu2psi(mu_t_lsm, K, r_cut, style)
# psi1_k_pos_lsm, psi2_k_pos_lsm = eigen2layer(K,r_cut,r1,r2,psi_k_pos_lsm,tau_k_pos_lsm,style)

# hk = params_est(K, dt, psi1_k_pos_lsm[:, :, :10000], psi2_k_pos_lsm[:, :, :10000], kd=10, beta=22, kappa=9, nu=1e-12, U=1, r_cut=16, style='circle')

data_path = '../data/qg/QG_DATA_topo40_nu1e-12_beta22_K128_dt1e-3_subs.mat'
with h5py.File(data_path, 'r') as file:
    print("Keys: %s" % file.keys())
    psi1_k_t = np.transpose(file['psi_1_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    psi2_k_t = np.transpose(file['psi_2_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
psi1_k_t = psi1_k_t['real'] + 1j * psi1_k_t['imag']
psi2_k_t = psi2_k_t['real'] + 1j * psi2_k_t['imag']
dt = dt*s_rate

hk = params_est(K, dt, psi1_k_t[:, :, :10000], psi2_k_t[:, :, :10000], kd=10, beta=22, kappa=9, nu=1e-12, U=1, r_cut=16, style='circle')

np.save('../data/hk_truth.npy', hk)