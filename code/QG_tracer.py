import numpy as np
from numba import jit
from QG import QG
from Lagrangian_tracer import Lagrange_tracer_model
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
            X, Y = np.meshgrid(np.arange(-np.pi, np.pi, dx), np.arange(-np.pi, np.pi, dx))
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
        x, y = tracer_model.forward_ens(ens, L, Nt, dt, x0, y0, psi_hat_history[:, :, :, :, 0])

        return psi_hat_history, x, y