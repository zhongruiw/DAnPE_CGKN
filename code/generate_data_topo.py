import numpy as np
from numba import jit, prange
from Lagrangian_tracer import Lagrange_tracer_model
from time import time
'''
require rocket-fft for numba to be aware of np.fft
!pip install rocket-fft 
'''

def generate_topo(N=128, alpha=4.0):
    '''generate a gaussian random field as topography'''
    kx = np.fft.fftfreq(N).reshape(-1, 1)
    ky = np.fft.fftfreq(N).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-6  # avoid division by zero
    spectrum = 1.0 / k**alpha # Power-law spectrum: 1 / k^alpha
    noise = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
    fft_field = noise * np.sqrt(spectrum)
    field = np.fft.ifft2(fft_field).real
    field -= np.mean(field)
    field /= np.std(field)
    return field

@jit(nopython=True)
def rhs_spectral_topo(q_hat, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian):
    q_vec = q_hat - subtract_hk
    # psi_hat = (q2psi @ q_vec[:,:,:,None])[:,:,:,0]
    # if using numba, manually do the matrix multiplication (2x2 @ 2x1)
    psi_hat = np.zeros((K,K,2), dtype=np.complex128)
    psi_hat[:, :, 0] = q2psi[:, :, 0, 0] * q_vec[:, :, 0] + q2psi[:, :, 0, 1] * q_vec[:, :, 1]
    psi_hat[:, :, 1] = q2psi[:, :, 1, 0] * q_vec[:, :, 0] + q2psi[:, :, 1, 1] * q_vec[:, :, 1]

    # Calculate Ekman plus beta plus mean shear
    RHS = np.zeros_like(q_hat, dtype=np.complex64)
    RHS[:,:,0] = -Ut * dX * q_hat[:,:,0] - (kb**2 + Ut * kd**2) * dX * psi_hat[:,:,0]
    RHS[:,:,1] = Ut * dX * q_hat[:,:,1] - (kb**2 - Ut * kd**2) * dX * psi_hat[:,:,1] - r * Laplacian * psi_hat[:,:,1]

    return RHS
    
@jit(nopython=True, parallel=True, fastmath=True)
def forward_loop(ens, qp_ens, Nt, dt, HV, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian, sigma_q):
    qp_ens_history = np.zeros((ens, Nt+1, K, K, 2))
    qp_ens_history[:, 0, :, :, :] = qp_ens
    q_ens = np.fft.fft2(qp_ens, axes=(1,2))
    # Timestepping
    for e in prange(ens):  # Parallelize over ensemble members
        q = q_ens[e, :, :, :]
        # Timestepping 
        for n in range(1, Nt+1):
            # # (Fourth-order Runge=Kutta pseudo-spectral scheme)
            # M = 1 / (1 - .25 * dt * HV)
            # # First stage ARK4
            # k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l0 = HV * q
            # # Second stage
            # q1 = M * (q + .5 * dt * k0 + .25 * dt * l0)
            # k1 = rhs_spectral_topo(q1, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l1 = HV * q1
            # # Third stage
            # q2 = M * (q + dt * (13861 * k0 / 62500 + 6889 * k1 / 62500 + 8611 * l0 / 62500 - 1743 * l1 / 31250))
            # k2 = rhs_spectral_topo(q2, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l2 = HV * q2
            # # Fourth stage
            # q3 = M * (q + dt * (-0.04884659515311858 * k0 - 0.1777206523264010 * k1 + 0.8465672474795196 * k2 +
            #                     0.1446368660269822 * l0 - 0.2239319076133447 * l1 + 0.4492950415863626 * l2))
            # k3 = rhs_spectral_topo(q3, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l3 = HV * q3
            # # Fifth stage
            # q4 = M * (q + dt * (-0.1554168584249155 * k0 - 0.3567050098221991 * k1 + 1.058725879868443 * k2 +
            #                     0.3033959883786719 * k3 + 0.09825878328356477 * l0 - 0.5915442428196704 * l1 +
            #                     0.8101210538282996 * l2 + 0.2831644057078060 * l3))
            # k4 = rhs_spectral_topo(q4, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l4 = HV * q4
            # # Sixth stage
            # q5 = M * (q + dt * (0.2014243506726763 * k0 + 0.008742057842904184 * k1 + 0.1599399570716811 * k2 +
            #                     0.4038290605220775 * k3 + 0.2260645738906608 * k4 + 0.1579162951616714 * l0 +
            #                     0.1867589405240008 * l2 + 0.6805652953093346 * l3 - 0.2752405309950067 * l4))
            # k5 = rhs_spectral_topo(q5, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            # l5 = HV * q5

            # # Successful step, proceed to evaluation
            # qp = np.real(np.fft.ifft2(q + dt * (0.1579162951616714 * (k0 + l0) +
            #                                     0.1867589405240008 * (k2 + l2) +
            #                                     0.6805652953093346 * (k3 + l3) -
            #                                     0.2752405309950067 * (k4 + l4) +
            #                                     (k5 + l5) / 4), axes=(0,1)))

            # (Foward Euler scheme)
            k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, q2psi, Ut, dX, dY, Laplacian)
            l0 = HV * q
            qp = np.real(np.fft.ifft2(q + dt * (k0 + l0), axes=(0,1)))
            # add stochastic noise
            qp += sigma_q * np.random.randn(K, K, 2) * np.sqrt(dt)

            q = np.fft.fft2(qp, axes=(0,1))
            qp_ens_history[e, n, :, :, :] = qp

    return qp_ens_history

class LinQG:
    '''
    Linearized stochastic QG
    '''
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None, sigma_q=0.1):
        # Set up hyperviscous PV dissipation
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        HV = np.tile((-nu * (KX**2 + KY**2)**4)[:,:, None], (1,1,2))

        # Initialize topography
        if topo is None:
            dx = 2 * np.pi / K
            X, Y = np.meshgrid(np.arange(0, 2*np.pi, dx), np.arange(0, 2*np.pi, dx))
            topo = H * (np.cos(X) + 2 * np.cos(2 * Y))
            topo -= np.mean(topo)  # subtracting the mean to center the topography
        hk = np.fft.fft2(topo)

        # Initialize additional variables for the simulation
        Ut = U  # zonal shear flow
        dX = 1j * KX
        dY = 1j * KY
        Laplacian = dX**2 + dY**2
        K_square = KX**2 + KY**2
        q2psi = -1 / (K_square * (K_square + kd**2))[:,:,None,None] * (np.eye(2) * K_square[:, :, None, None] + kd**2/2)
        q2psi[0,0,:,:] = 0
        q2psi = q2psi.astype(np.complex128)
        subtract_hk = np.zeros(((K,K,2)), dtype=complex)
        subtract_hk[:,:,1] = hk

        self.K = K
        self.kd = kd
        self.kb = kb
        self.r = r
        self.hk = hk
        self.dX = dX
        self.dY = dY
        self.q2psi = q2psi
        self.HV = HV
        self.Ut = Ut
        self.Laplacian = Laplacian
        self.subtract_hk = subtract_hk
        self.sigma_q = sigma_q
            
    def forward_ens(self, ens, Nt=1, dt=1e-3, qp_ens=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        '''
        return forward_loop(ens, qp_ens, Nt, dt, self.HV, self.K, self.kd, self.kb, self.r, self.subtract_hk, self.q2psi, self.Ut, self.dX, self.dY, self.Laplacian, self.sigma_q)

class QG_tracer:
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None, sigma_xy=0.1, style='square', sigma_q=0.1):
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

        self.flow_model = LinQG(K, kd, kb, U, r, nu, H, topo, sigma_q)
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

        # run tracer model
        x, y = tracer_model.forward_ens(ens, L, Nt, dt, x0, y0, psi_hat_history[:, :, :, :, 0]) # of shape (ens,L,Nt+1)

        return psi_hat_history, x, y, qp_history

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
    np.random.seed(9)

    # ---------- QG model parameters ------------
    K = 64 # Number of points (also Fourier modes) in each direction
    kd = 10 # Nondimensional deformation wavenumber
    kb = np.sqrt(22) # Nondimensional beta wavenumber, beta = kb^2 
    U = 1 # Zonal shear flow
    r = 9 # Nondimensional Ekman friction coefficient
    nu = 1e-12 # Coefficient of biharmonic vorticity diffusion
    H = 40 # Topography parameter
    dt = 2e-3 # Time step size
    warm_up = 1000 # Warm-up time steps
    Nt = 5e3 + warm_up # Number of time steps
    num_topo = 100 # Number of simulations to generate with various topography

    # ------- Tracer observation parameters -------
    L = 128 # Number of tracers
    sigma_xy = 0.1 # Tracer observation noise std (in sde)
    dt_ob = 4e-2 # Observation time interval
    obs_freq = int(dt_ob / dt) # Observation frequency
    Nt_obs = int((Nt - warm_up) / obs_freq + 1) # Number of observations saved

    # ---------------- Simulations ----------------
    topos = np.zeros((num_topo, K, K))
    xy_truths = np.zeros((num_topo, Nt_obs, L, 2))
    xy_obss = np.zeros((num_topo, Nt_obs, L, 2))
    t0 = time()
    for n_topo in prange(num_topo):
        # ------------ initialization ------------
        topo = generate_topo(K, 4) # generate topography
        qp = np.zeros((K, K, 2))
        qp[:, :, 1] = 10 * np.random.randn(K, K)
        qp[:, :, 1] -= np.mean(qp[:, :, 1])
        qp[:, :, 0] = qp[:, :, 1]
        model = QG_tracer(K=K, kd=kd, kb=kb, U=U, r=r, nu=nu, topo=H*topo, sigma_xy=sigma_xy)
        x_t = np.zeros((Nt_obs, L))
        y_t = np.zeros((Nt_obs, L))
        x0 = np.pi + 0.1 * np.random.randn(L) # np.random.uniform(0, 2*np.pi, L)
        y0 = np.pi + 0.1 * np.random.randn(L) # np.random.uniform(0, 2*np.pi, L)
        x_t[0, :] = x0
        y_t[0, :] = y0

        # warm up
        qp_t, psi_k_t1 = model.forward_flow(ens=1, Nt=warm_up, dt=dt, qp_ens=qp[None,:,:,:])
        qp_t0 = qp_t[:, -1, :, :, :]

        for i in range(1, Nt_obs):
            psi_k_t1, x1, y1, qp_t1 = model.forward_ens(ens=1, Nt=obs_freq, dt=dt, qp_ens=qp_t0, L=L, x0=x0[None, :], y0=y0[None, :])
            x_t[i, :] = x1[0, :, -1]
            y_t[i, :] = y1[0, :, -1]
            x0 = x1[0, :, -1]
            y0 = y1[0, :, -1]
            qp_t0 = qp_t1[:, -1, :, :, :]

        xy_truth = np.concatenate((x_t[:,:,None], y_t[:,:,None]), axis=2)
        sigma_obs = 0.01
        xy_obs = xy_truth + sigma_obs * np.random.randn(xy_truth.shape[0], xy_truth.shape[1], xy_truth.shape[2])
        xy_obs = np.mod(xy_obs, 2*np.pi)  # Periodic boundary conditions

        topos[n_topo] = H*topo
        xy_truths[n_topo] = xy_truth
        xy_obss[n_topo] = xy_obs

    t1 = time()
    print('time used: {:.2f} hours'.format((t1-t0)/3600))

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
    'xy_obs': xy_obss,
    'sigma_obs': sigma_obs,
    'xy_truth': xy_truths,
    'sigma_xy': sigma_xy,
    }
    np.savez('../data/tracer_data_topo_rndseed9.npz', **save)

