import numpy as np
from numba import jit
from QG import QG
from Lagrangian_tracer import Lagrange_tracer_model
from time import time

'''
require rocket-fft for numba to be aware of np.fft
!pip install rocket-fft 
'''

class QG_tracer:
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None, sigma_xy=0.1, style='square'):
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)

        # Initialize topography
        if topo is None:
            dx = 2 * np.pi / K
            X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
            topo = H * (np.cos(X) + 2 * np.cos(2 * Y))
            topo -= np.mean(topo)  # subtracting the mean to center the topography
        hk = np.fft.fft2(topo)

        # Initialize additional variables for the simulation
        K_square = KX**2 + KY**2
        q2psi = -1 / (K_square * (K_square + kd**2))[:,:,None,None] * (np.eye(2) * K_square[:, :, None, None] + kd**2/2)
        q2psi[0,0,:,:] = 0
        q2psi = q2psi.astype(np.complex128)
        subtract_hk = np.zeros(((K,K,2)), dtype=complex)
        subtract_hk[:,:,1] = hk

        self.flow_model = QG(K, kd, kb, U, r, nu, H, topo)
        self.tracer_model = Lagrange_tracer_model(K, sigma_xy, style)
        self.q2psi = q2psi
        self.subtract_hk = subtract_hk

    def forward_ens(self, ens, Nt=1, dt=1e-3, qp_ens=None, L=1, x0=None, y0=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        x0, y0: arrays of shape (ens, L)
        '''
        K = qp_ens.shape[1]
        flow_model = self.flow_model
        tracer_model = self.tracer_model
        q2psi = self.q2psi
        subtract_hk = self.subtract_hk

        # run flow model
        qp_history = flow_model.forward_ens(ens, Nt, dt, qp_ens)
        q_hat_history = np.fft.fft2(qp_history, axes=(2,3))

        # q to psi
        q_vec = q_hat_history - subtract_hk
        psi_hat_history = (q2psi @ q_vec[:,:,:,:,:,None])[:,:,:,:,:,0] # of shape (ens,Nt+1,K,K,2)

        # # psi to velocity
        # u = np.real(np.fft.ifft2(psi_hat_history[:, -1, :, :, :] * 1j * KY[:, :, None], axes=(1,2)))
        # v = np.real(np.fft.ifft2(psi_hat_history[:, -1, :, :, :] * (-1j) * KX[:, :, None], axes=(1,2)))

        # run tracer model
        x, y = tracer_model.forward_ens(ens, L, Nt, dt, x0, y0, psi_hat_history[:, :, :, :, 0]) # of shape (ens,L,Nt+1)

        return psi_hat_history, x, y, qp_history#, u, v

    def forward_flow(self, ens=1, Nt=1, dt=1e-3, qp_ens=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        '''
        flow_model = self.flow_model
        q2psi = self.q2psi
        subtract_hk = self.subtract_hk

        # run flow model
        qp_history = flow_model.forward_ens(ens, Nt, dt, qp_ens) # of shape (ens, Nt+1, K, K, 2)
        q_hat_history = np.fft.fft2(qp_history, axes=(2,3))

        # q to psi
        q_vec = q_hat_history - subtract_hk
        psi_hat_history = (q2psi @ q_vec[:,:,:,:,:,None])[:,:,:,:,:,0] # of shape (ens,Nt+1,K,K,2)

        return qp_history, psi_hat_history


if __name__ == "__main__": 
    np.random.seed(1)

    # ---------- QG model parameters ------------
    K = 128 # Number of points (also Fourier modes) in each direction
    kd = 10 # Nondimensional deformation wavenumber
    kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
    U = 1 # Zonal shear flow
    r = 9 # Nondimensional Ekman friction coefficient
    nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
    H = 40 # Topography parameter
    dt = 2e-3 # Time step size
    warm_up = 1000 # Warm-up time steps
    Nt = 500 + warm_up # Number of time steps

    # ------- Tracer observation parameters -------
    L = 128 # Number of tracers
    sigma_xy = 0.1 # Tracer observation noise std (in sde)
    dt_ob = 2e-2 # Observation time interval
    obs_freq = int(dt_ob / dt) # Observation frequency
    Nt_obs = int((Nt - warm_up) / obs_freq + 1) # Number of observations saved

    # number of simulations
    N_ens = 200

    # -------------- initialization ---------------
    qp = np.zeros((N_ens, K, K, 2))
    qp[:, :, :, 1] = 10 * np.random.randn(N_ens, K, K)
    qp[:, :, :, 1] -= np.mean(qp[:, :, :, 1])
    qp[:, :, :, 0] = qp[:, :, :, 1]
    psi_k_t = np.zeros((N_ens, Nt_obs, K, K, 2), dtype=complex)
    model = QG_tracer(K=K, kd=kd, kb=kb, U=U, r=r, nu=nu, H=H, sigma_xy=sigma_xy)
    x_t = np.zeros((N_ens, Nt_obs, L))
    y_t = np.zeros((N_ens, Nt_obs, L))
    x0 = np.pi + 0.1 * np.random.randn(N_ens, L) # np.random.uniform(0, 2*np.pi, L)
    y0 = np.pi + 0.1 * np.random.randn(N_ens, L) # np.random.uniform(0, 2*np.pi, L)
    x_t[:, 0, :] = x0
    y_t[:, 0, :] = y0

    t0 = time()

    # warm up
    qp_t, psi_k_t1 = model.forward_flow(ens=N_ens, Nt=warm_up, dt=dt, qp_ens=qp[:,:,:,:])
    qp_t0 = qp_t[:, -1, :, :, :]
    psi_k_t[:, 0, :, :, :] = psi_k_t1[:, -1, :, :, :]

    # ------------ model integration --------------
    for i in range(1, Nt_obs):
        psi_k_t1, x1, y1, qp_t1 = model.forward_ens(ens=N_ens, Nt=obs_freq, dt=dt, qp_ens=qp_t0, L=L, x0=x0[:, :], y0=y0[:, :])
        psi_k_t[:, i, :, :, :] = psi_k_t1[:, -1, :, :, :]
        x_t[:, i, :] = x1[:, :, -1]
        y_t[:, i, :] = y1[:, :, -1]
        x0 = x1[:, :, -1]
        y0 = y1[:, :, -1]
        qp_t0 = qp_t1[:, -1, :, :, :]

    psi_t = np.fft.ifft2(psi_k_t, axes=(2,3))

    t1 = time()
    print('time used: {:.2f} hours'.format((t1-t0)/3600))

    # check imaginary part
    max_imag_abs = np.max(np.abs(np.imag(psi_t)))
    if max_imag_abs > 1e-8:
        raise Exception("get significant imaginary parts, check ifft2")
    else:
        psi_t = np.real(psi_t)

    xy_truth = np.concatenate((x_t[:,:,:,None], y_t[:,:,:,None]), axis=3)
    sigma_obs = 0.01
    xy_obs = xy_truth + sigma_obs * np.random.randn(xy_truth.shape[0], xy_truth.shape[1], xy_truth.shape[2], xy_truth.shape[3])
    xy_obs = np.mod(xy_obs, 2*np.pi)  # Periodic boundary conditions
    sigma_psi = 0.01
    psi_noisy = psi_t + sigma_psi * np.random.randn(psi_t.shape[0], psi_t.shape[1], psi_t.shape[2], psi_t.shape[3], psi_t.shape[4])

    save = {
    'K': K,
    'kd': kd,
    'kb': kb,
    'U': U,
    'r': r,
    'H': H,
    'nu': nu,
    'dt': dt,
    'L': L,
    'dt_ob': dt_ob,
    'xy_obs': xy_obs,
    'psi_noisy': psi_noisy,
    'sigma_obs': sigma_obs,
    'sigma_psi': sigma_psi,
    'psi_truth': psi_t,
    'xy_truth': xy_truth,
    'sigma_xy': sigma_xy,
    }
    np.savez('../data/qg_data_sigmaxy01_continuousxy_1.npz', **save)

