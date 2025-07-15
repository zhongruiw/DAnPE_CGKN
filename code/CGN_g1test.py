import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector
import time

def unit2xy(xy_unit):
    cos0 = xy_unit[..., 0]
    sin0 = xy_unit[..., 1]
    cos1 = xy_unit[..., 2]
    sin1 = xy_unit[..., 3]
    x = torch.atan2(sin0, cos0) # range [-pi, pi)
    y = torch.atan2(sin1, cos1) # range [-pi, pi)

    return x, y

# class Field2point(torch.nn.Module):
#     def __init__(self, K, dtype=torch.float32, chunk_size=32):
#         super().__init__()
#         self.K = K
#         self.dtype = dtype
#         self.chunk_size = chunk_size

#         kx = torch.fft.fftfreq(K, d=1.0/K).to(dtype)
#         ky = torch.fft.fftfreq(K, d=1.0/K).to(dtype)
#         KX, KY = torch.meshgrid(kx, ky, indexing="xy")
#         self.register_buffer("KX_flat", KX.flatten().to(dtype=torch.complex64))  # (K²,)
#         self.register_buffer("KY_flat", KY.flatten().to(dtype=torch.complex64))  # (K²,)

#     def forward(self, x0, y0, psi):
#         """
#         Parameters:
#         - x0, y0: (B, L) float tensors of tracer positions (range [0, 2pi])
#         - psi: (B, K, K) real-valued streamfunction fields

#         Returns:
#         - psi_x: (B, L) interpolated psi values at the positions
#         """

#         B, L = x0.shape
#         K, KX_flat, KY_flat = self.K, self.KX_flat, self.KY_flat  # (K²,)

#         # FFT on each sample in the batch
#         psi_hat = torch.fft.fft2(psi)  # (B, K, K)
#         psi_hat_flat = psi_hat.reshape(B, -1)  # (B, K²)
#         psi_x = torch.empty(B, L, dtype=self.dtype, device=psi.device)
#         for start in range(0, L, self.chunk_size):
#             end = min(start + self.chunk_size, L)
#             x_chunk = x0[:, start:end]  # (B, chunk)
#             y_chunk = y0[:, start:end]  # (B, chunk)

#             # (B, chunk, K²) = (B, chunk, 1) × (1, 1, K²)
#             exp_term = torch.exp(1j * (x_chunk[..., None] * KX_flat + y_chunk[..., None] * KY_flat))  # (B, chunk, K²)
#             psi_chunk = torch.einsum("blk,bk->bl", exp_term, psi_hat_flat)  # (B, chunk)
#             psi_x[:, start:end] = torch.real(psi_chunk) / (K ** 2)

#         return psi_x  # shape (B, L)

class Field2point(torch.nn.Module):
    def __init__(self, K, dtype=torch.float32, chunk_size=32):
        super().__init__()
        self.K = K
        self.dtype = dtype
        self.chunk_size = chunk_size

        kx = torch.fft.fftshift(torch.fft.fftfreq(K, d=1.0/K).to(dtype))
        ky = torch.fft.fftshift(torch.fft.fftfreq(K, d=1.0/K).to(dtype))
        KX, KY = torch.meshgrid(kx, ky, indexing="xy")
        self.register_buffer("KX_flat", KX.flatten().to(dtype=torch.complex64))  # (K²,)
        self.register_buffer("KY_flat", KY.flatten().to(dtype=torch.complex64))  # (K²,)

    def forward(self, x0, y0, psi):
        """
        Parameters:
        - x0, y0: (B, L) float tensors of tracer positions (range [-pi, pi])
        - psi: (B, K, K) real-valued streamfunction fields

        Returns:
        - psi_x: (B, L) interpolated psi values at the positions
        """

        B, L = x0.shape
        K, KX_flat, KY_flat = self.K, self.KX_flat, self.KY_flat  # (K²,)

        # FFT on each sample in the batch
        psi_hat = torch.fft.fftshift(torch.fft.fft2(psi), dim=(-2, -1))  # (B, K, K)
        psi_hat_flat = psi_hat.reshape(B, -1)  # (B, K²)
        psi_x = torch.empty(B, L, dtype=self.dtype, device=psi.device)
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            x_chunk = x0[:, start:end]  # (B, chunk)
            y_chunk = y0[:, start:end]  # (B, chunk)

            # (B, chunk, K²) = (B, chunk, 1) × (1, 1, K²)
            exp_term = torch.exp(1j * (x_chunk[..., None] * KX_flat + y_chunk[..., None] * KY_flat))  # (B, chunk, K²)
            psi_chunk = torch.einsum("blk,bk->bl", exp_term, psi_hat_flat)  # (B, chunk)
            psi_x[:, start:end] = torch.real(psi_chunk) / (K ** 2)

        return psi_x  # shape (B, L)

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_sigmaxy01_t1000_subsampled.npz")

pos = torch.tensor(QG_Data["xy_obs"], dtype=torch.float) # shape (Nt, L, 2)
psi = torch.tensor(QG_Data["psi_noisy"][..., 0], dtype=torch.float) # shape (Nt, Nx, Nx)
pos_unit = torch.stack([torch.cos(pos[:, :, 0]), torch.sin(pos[:, :, 0]), torch.cos(pos[:, :, 1]), torch.sin(pos[:, :, 1])], axis=-1)
pos = pos.to(device)
psi = psi.to(device)
pos_unit = pos_unit.to(device)

field2point = Field2point(K=64, chunk_size=2).to(device)
with torch.no_grad():
    psi_pos = field2point(pos[..., 0], pos[..., 1], psi) # shape (Nt, L)

# Train / Test
Ntrain = 20000
Ntest = 5000

# train_pos = pos[:Ntrain]
train_pos_unit = pos_unit[:Ntrain]
train_psi = psi[:Ntrain]
train_psi_pos = psi_pos[:Ntrain]
# test_pos = pos[Ntrain:Ntrain+Ntest]
test_pos_unit = pos_unit[Ntrain:Ntrain+Ntest]
test_psi = psi[Ntrain:Ntrain+Ntest]
test_psi_pos = psi_pos[Ntrain:Ntrain+Ntest]

############################################################
###########################  CGN  ##########################
############################################################
class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.g1_size = (dim_u1, dim_z)
        self.use_pos_encoding = use_pos_encoding
        self.num_frequencies = num_frequencies
        # Homogeneous Tracers
        in_dim = 2  # (x1, x2)
        if use_pos_encoding:
            in_dim = 2 + 2 * 2 * num_frequencies  # sin/cos for each x1 and x2
        self.output_size = dim_z
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(), 
                                 nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(),
                                 # nn.Linear(16, 32), nn.SiLU(),
                                 # nn.Linear(32, 64), nn.SiLU(),
                                 # nn.Linear(64, 32), nn.SiLU(),
                                 # nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(128, self.output_size))

    def forward(self, x, z): # x shape(Nt, L, 2), z shape (Nt, dim_z)
        Nt, L = x.shape[:2]
        if self.use_pos_encoding:
            x = self.positional_encoding(x.view(-1,2)) # (Nt * L, in_dim)
            x = x.view(Nt, L, -1)

        g1 = self.net(x)    # (Nt, L, dimz)
        g1 = nnF.softmax(g1 / 1, dim=-1) # sharp attention
        z = z.unsqueeze(-1) # (Nt, dim_z, 1)
        z_pos_pred = g1@z   # (Nt, L, 1)
        return z_pos_pred.squeeze(-1) # (Nt, L)

    def positional_encoding(self, x):
        """
        Input: x of shape (B, 2)
        Output: encoded x of shape (B, 2+ 4*num_frequencies)
        """
        freqs = 2 ** torch.arange(self.num_frequencies, device=x.device) * np.pi  # (F,)
        # freqs = torch.logspace(0, np.log10(np.pi * 2**self.num_frequencies), self.num_frequencies, device=x.device)
        x1 = x[:, 0:1] * freqs                 # (B, F)
        x2 = x[:, 1:2] * freqs                 # (B, F)
        encoded = torch.cat([x, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)  # shape: (B, 2+4F)
        return encoded

########################################################
####################### Train cgn ######################
########################################################
dim_u1 = 128  # L
dim_u2 = 64*64
dim_z = 64*64

epochs = 500
train_batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_pos_unit, train_psi, train_psi_pos)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history = []

cgn = CGN(dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6).to(device)
optimizer = torch.optim.Adam(cgn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
for ep in range(1, epochs+1):
    cgn.train()
    start_time = time.time()
    train_loss = 0.
    for pos_unit_batch, psi_batch, psi_pos_batch in train_loader:
        pos_unit_batch, psi_batch, psi_pos_batch = pos_unit_batch.to(device), psi_batch.to(device), psi_pos_batch.to(device)

        # Prediction
        psi_batch_flat = psi_batch.view(-1, dim_z)    # (B, dim_z)
        x_batch, y_batch = unit2xy(pos_unit_batch)
        pos_batch = torch.stack([x_batch, y_batch], dim=-1)
        psi_pos_pred = cgn(pos_batch, psi_batch_flat) # (B, L)

        loss_total = nnF.mse_loss(psi_pos_batch, psi_pos_pred)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss_total.item()

    train_loss /= train_num_batches
    train_loss_history.append(train_loss)
    print("ep", ep,
          " time:", round(time.time() - start_time, 2),
          " loss:", round(train_loss, 6))

torch.save(cgn, r"../model/QG(Noisy)_64x64_sigmaxy01_CGN_g1test.pt")
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGN_g1test_train_loss_history.npy", train_loss_history)

# """


############################################
################# Test cgn #################
############################################
# cgkn = torch.load(r"../model/QG(Noisy)_64x64_sigmaxy01_CGN_g1test.pt").to(device)
cgn.eval()

batch_size = 200
test_pos = test_pos.to(device)
test_psi = test_psi.to(device)
test_psi_pos = test_psi_pos.to(device)
test_psi_pos_preds = []
with torch.no_grad():
    for i in range(0, Ntest, batch_size):
        batch_pos = test_pos[i:i+batch_size]
        batch_psi = test_psi[i:i+batch_size]
        batch_psi_flat = batch_psi.view(-1, dim_z)          # (B, dim_z)
        batch_psi_pos_pred = cgn(batch_pos, batch_psi_flat) # (B, L)
        test_psi_pos_preds.append(batch_psi_pos_pred)
    test_psi_pos_pred = torch.cat(test_psi_pos_preds, dim=0)

MSE = nnF.mse_loss(test_psi_pos, test_psi_pos_pred)
print("Test MSE:", MSE.item())
np.save(r"../data/CGN_64x64_g1_test.npy", test_psi_pos_pred.to("cpu"))

# Number of Parameters
cgn_params = parameters_to_vector(cgn.parameters()).numel()
# encoder_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
# decoder_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
total_params = cgn_params #+ encoder_params + decoder_params
print(f'cgn #parameters:      {cgn_params:,}')
# print(f'encoder #parameters:  {encoder_params:,}')
# print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')
