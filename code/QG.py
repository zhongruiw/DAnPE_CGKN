import numpy as np
from numba import jit, prange
from tqdm import tqdm
'''
require rocket-fft for numba to be aware of np.fft
!pip install rocket-fft 
'''

@jit(nopython=True)
def rhs_spectral_topo(q_hat, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian):
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

    # For using a 3/2-rule dealiased jacobian
    Psi_hat = np.zeros((int(1.5 * K), int(1.5 * K), 2), dtype=np.complex128)
    Q_hat = np.zeros((int(1.5 * K), int(1.5 * K), 2), dtype=np.complex128)
    Psi_hat[:K//2+1, :K//2+1, :] = (9/4) * psi_hat[:K//2+1, :K//2+1, :]
    Psi_hat[:K//2+1, K+1:int(1.5*K), :] = (9/4) * psi_hat[:K//2+1, K//2+1:K, :]
    Psi_hat[K+1:int(1.5*K), :K//2+1, :] = (9/4) * psi_hat[K//2+1:K, :K//2+1, :]
    Psi_hat[K+1:int(1.5*K), K+1:int(1.5*K), :] = (9/4) * psi_hat[K//2+1:K, K//2+1:K, :]
    Q_hat[:K//2+1, :K//2+1, :] = (9/4) * q_hat[:K//2+1, :K//2+1, :]
    Q_hat[:K//2+1, K+1:int(1.5*K), :] = (9/4) * q_hat[:K//2+1, K//2+1:K, :]
    Q_hat[K+1:int(1.5*K), :K//2+1, :] = (9/4) * q_hat[K//2+1:K, :K//2+1, :]
    Q_hat[K+1:int(1.5*K), K+1:int(1.5*K), :] = (9/4) * q_hat[K//2+1:K, K//2+1:K, :]

    # Calculate u.gradq on 3/2 grid
    u = np.real(np.fft.ifft2(-DY[:,:,None] * Psi_hat, axes=(0,1)))
    v = np.real(np.fft.ifft2(DX[:,:,None] * Psi_hat, axes=(0,1)))
    qx = np.real(np.fft.ifft2(DX[:,:,None] * Q_hat, axes=(0,1)))
    qy = np.real(np.fft.ifft2(DY[:,:,None] * Q_hat, axes=(0,1)))
    jaco_real = u * qx + v * qy

    # FFT, 3/2 grid; factor of (4/9) scales fft
    Jaco_hat = (4/9) * np.fft.fft2(jaco_real, axes=(0,1))

    # Reduce to normal grid
    jaco_hat = np.zeros((K, K, 2), dtype=np.complex128)
    jaco_hat[:K//2 + 1, :K//2 + 1, :] = Jaco_hat[:K//2 + 1, :K//2 + 1, :]
    jaco_hat[:K//2 + 1, K//2 + 1:, :] = Jaco_hat[:K//2 + 1, K+1:int(1.5 * K), :]
    jaco_hat[K//2 + 1:, :K//2 + 1, :] = Jaco_hat[K+1:int(1.5 * K), :K//2 + 1, :]
    jaco_hat[K//2 + 1:, K//2 + 1:, :] = Jaco_hat[K+1:int(1.5 * K), K+1:int(1.5 * K), :]

    # Put it all together
    RHS -= jaco_hat

    return RHS
    
@jit(nopython=True, parallel=True, fastmath=True)
def forward_loop(ens, qp_ens, Nt, dt, HV, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian):
    qp_ens_history = np.zeros((ens, Nt+1, K, K, 2))
    qp_ens_history[:, 0, :, :, :] = qp_ens
    q_ens = np.fft.fft2(qp_ens, axes=(1,2))
    # Timestepping
    for e in prange(ens):  # Parallelize over ensemble members
        q = q_ens[e, :, :, :]
        # Timestepping (Fourth-order Runge=Kutta pseudo-spectral scheme)
        for n in range(1, Nt+1):
            M = 1 / (1 - .25 * dt * HV)
            # First stage ARK4
            k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l0 = HV * q
            # Second stage
            q1 = M * (q + .5 * dt * k0 + .25 * dt * l0)
            k1 = rhs_spectral_topo(q1, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l1 = HV * q1
            # Third stage
            q2 = M * (q + dt * (13861 * k0 / 62500 + 6889 * k1 / 62500 + 8611 * l0 / 62500 - 1743 * l1 / 31250))
            k2 = rhs_spectral_topo(q2, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l2 = HV * q2
            # Fourth stage
            q3 = M * (q + dt * (-0.04884659515311858 * k0 - 0.1777206523264010 * k1 + 0.8465672474795196 * k2 +
                                0.1446368660269822 * l0 - 0.2239319076133447 * l1 + 0.4492950415863626 * l2))
            k3 = rhs_spectral_topo(q3, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l3 = HV * q3
            # Fifth stage
            q4 = M * (q + dt * (-0.1554168584249155 * k0 - 0.3567050098221991 * k1 + 1.058725879868443 * k2 +
                                0.3033959883786719 * k3 + 0.09825878328356477 * l0 - 0.5915442428196704 * l1 +
                                0.8101210538282996 * l2 + 0.2831644057078060 * l3))
            k4 = rhs_spectral_topo(q4, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l4 = HV * q4
            # Sixth stage
            q5 = M * (q + dt * (0.2014243506726763 * k0 + 0.008742057842904184 * k1 + 0.1599399570716811 * k2 +
                                0.4038290605220775 * k3 + 0.2260645738906608 * k4 + 0.1579162951616714 * l0 +
                                0.1867589405240008 * l2 + 0.6805652953093346 * l3 - 0.2752405309950067 * l4))
            k5 = rhs_spectral_topo(q5, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)
            l5 = HV * q5

            # Successful step, proceed to evaluation
            qp = np.real(np.fft.ifft2(q + dt * (0.1579162951616714 * (k0 + l0) +
                                                0.1867589405240008 * (k2 + l2) +
                                                0.6805652953093346 * (k3 + l3) -
                                                0.2752405309950067 * (k4 + l4) +
                                                (k5 + l5) / 4), axes=(0,1)))
            q = np.fft.fft2(qp, axes=(0,1))

            qp_ens_history[e, n, :, :, :] = qp

    return qp_ens_history
    

class QG:
    def __init__(self, K=128, kd=10, kb=np.sqrt(22), U=1, r=9, nu=1e-12, H=40, topo=None):
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

        # for 3/2-rule dealiased Jacobian
        K_Jac = int(K * 3/2)
        Kx_Jac = np.fft.fftfreq(K_Jac) * K_Jac
        Ky_Jac = np.fft.fftfreq(K_Jac) * K_Jac
        KX_Jac, KY_Jac = np.meshgrid(Kx_Jac, Ky_Jac)
        DX = 1j * KX_Jac
        DY = 1j * KY_Jac

        self.K = K
        self.kd = kd
        self.kb = kb
        self.r = r
        self.hk = hk
        self.dX = dX
        self.dY = dY
        self.DX = DX
        self.DY = DY
        self.q2psi = q2psi
        self.HV = HV
        self.Ut = Ut
        self.Laplacian = Laplacian
        self.subtract_hk = subtract_hk

    def forward(self, Nt=10000, dt=1e-3, qp=None):
        K = self.K
        HV = self.HV
        kd = self.kd
        r = self.r
        subtract_hk = self.subtract_hk
        DX  = self.DX
        DY = self.DY
        q2psi = self.q2psi
        Ut = self.Ut
        dX = self.dX
        dY = self.dY

        # Initialize potential vorticity
        if qp is None:
            qp = np.zeros((K, K, 2))
            qp[:, :, 1] = 10 * np.random.randn(K, K)
            qp[:, :, 1] -= np.mean(qp[:, :, 1])  # Centering the PV
            qp[:, :, 0] = qp[:, :, 1]
        q = np.fft.fft2(qp, axes=(0,1))

        qp_history = np.zeros((Nt+1, K, K, 2))
        qp_history[0, :, :, :] = qp
        # Timestepping (Fourth-order Runge=Kutta pseudo-spectral scheme)
        for ii in tqdm(range(1, Nt+1), desc='Timestepping'):
            M = 1 / (1 - .25 * dt * HV)
            # First stage ARK4
            k0 = rhs_spectral_topo(q, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l0 = HV * q
            # Second stage
            q1 = M * (q + .5 * dt * k0 + .25 * dt * l0)
            k1 = rhs_spectral_topo(q1, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l1 = HV * q1
            # Third stage
            q2 = M * (q + dt * (13861 * k0 / 62500 + 6889 * k1 / 62500 + 8611 * l0 / 62500 - 1743 * l1 / 31250))
            k2 = rhs_spectral_topo(q2, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l2 = HV * q2
            # Fourth stage
            q3 = M * (q + dt * (-0.04884659515311858 * k0 - 0.1777206523264010 * k1 + 0.8465672474795196 * k2 +
                                0.1446368660269822 * l0 - 0.2239319076133447 * l1 + 0.4492950415863626 * l2))
            k3 = rhs_spectral_topo(q3, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l3 = HV * q3
            # Fifth stage
            q4 = M * (q + dt * (-0.1554168584249155 * k0 - 0.3567050098221991 * k1 + 1.058725879868443 * k2 +
                                0.3033959883786719 * k3 + 0.09825878328356477 * l0 - 0.5915442428196704 * l1 +
                                0.8101210538282996 * l2 + 0.2831644057078060 * l3))
            k4 = rhs_spectral_topo(q4, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l4 = HV * q4
            # Sixth stage
            q5 = M * (q + dt * (0.2014243506726763 * k0 + 0.008742057842904184 * k1 + 0.1599399570716811 * k2 +
                                0.4038290605220775 * k3 + 0.2260645738906608 * k4 + 0.1579162951616714 * l0 +
                                0.1867589405240008 * l2 + 0.6805652953093346 * l3 - 0.2752405309950067 * l4))
            k5 = rhs_spectral_topo(q5, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY)
            l5 = HV * q5

            # Successful step, proceed to evaluation
            qp = np.real(np.fft.ifft2(q + dt * (0.1579162951616714 * (k0 + l0) +
                                                0.1867589405240008 * (k2 + l2) +
                                                0.6805652953093346 * (k3 + l3) -
                                                0.2752405309950067 * (k4 + l4) +
                                                (k5 + l5) / 4), axes=(0,1)))
            q = np.fft.fft2(qp, axes=(0,1))

            qp_history[ii, :, :, :] = qp

        return qp_history
            
    def forward_ens(self, ens, Nt=1, dt=1e-3, qp_ens=None):
        '''
        qp_ens: array of shape (ens, K, K, 2)
        '''
        K = self.K
        HV = self.HV
        kd = self.kd
        kb = self.kb
        r = self.r
        subtract_hk = self.subtract_hk
        DX  = self.DX
        DY = self.DY
        q2psi = self.q2psi
        Ut = self.Ut
        dX = self.dX
        dY = self.dY
        Laplacian = self.Laplacian

        return forward_loop(ens, qp_ens, Nt, dt, HV, K, kd, kb, r, subtract_hk, DX, DY, q2psi, Ut, dX, dY, Laplacian)