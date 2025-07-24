import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector
import time
from torchviz import make_dot

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_sigmaxy01_t2000.npz")

pos = QG_Data["xy_obs"]
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1)
psi = QG_Data["psi_noisy"]

# u1 = torch.tensor(pos_unit, dtype=torch.float).view(pos.shape[0], -1)
u1 = torch.tensor(pos_unit, dtype=torch.float) # shape (Nt, L, 4), keep tracers parallel
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 128, 128, 2)

# Train / Test
Ntrain = 40000
Ntest = 10000

train_u1 = u1[:Ntrain]
train_u2 = u2[:Ntrain]
test_u1 = u1[Ntrain:Ntrain+Ntest]
test_u2 = u2[Ntrain:Ntrain+Ntest]

############################################################
######################### Phyiscs ##########################
############################################################

def unit2xy(xy_unit):
    cos0 = xy_unit[..., 0]
    sin0 = xy_unit[..., 1]
    cos1 = xy_unit[..., 2]
    sin1 = xy_unit[..., 3]
    x = torch.atan2(sin0, cos0) # range [-pi, pi)
    y = torch.atan2(sin1, cos1) # range [-pi, pi)

    return x, y

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


############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################
class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # manual padding
            bias=True
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        H, W = x.shape[-2:]
        pad_h = self._compute_padding(H, self.kernel_size[0], self.stride[0])
        pad_w = self._compute_padding(W, self.kernel_size[1], self.stride[1])

        # Apply circular padding manually
        x = nnF.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='circular')
        x = self.conv(x)
        return x

    def _compute_padding(self, size, k, s):
        if size % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (size % s), 0)
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        return (pad_top, pad_bottom)

class CircularConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # we'll manually manage padding
            bias=True
        )
        # compute circular padding width
        self.pad_w = int(0.5 + (self.kernel_size[1] - 1.) / (2. * self.stride[1]))
        self.pad_h = int(0.5 + (self.kernel_size[0] - 1.) / (2. * self.stride[0]))

    def forward(self, x):
        # Apply circular padding manually before transposed convolution
        x_padded = nnF.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='circular')
        out = self.deconv(x_padded)

        # Crop the output (equivalent to cropping `crop=stride * pad`)
        crop_h = (self.stride[0]+1) * self.pad_h
        crop_w = (self.stride[1]+1) * self.pad_w
        out = out[:, :, crop_h:out.shape[2]-crop_h, crop_w:out.shape[3]-crop_w]
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            CircularConv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, u):                           # u:(B, H, W, 2)
        u = u.permute(0, 3, 1, 2)                   # → (B, 2, H, W)
        out = self.enc(u)                           # → (B, 2, d1, d2)
        return out#.squeeze(1)                      # → (B, 2, d1, d2)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            CircularConvTranspose2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z):                  # z: (B, 2, d1, d2)
        u = self.dec(z)                    # (B, 2, 64, 64)
        return u.permute(0, 2, 3, 1)       # (B, 64, 64, 2)
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.f1_size = (dim_u1, 1)
        self.g1_size = (dim_u1, dim_z)
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.use_pos_encoding = use_pos_encoding
        self.num_frequencies = num_frequencies
        # Homogeneous Tracers
        in_dim = 2  # (x1, x2)
        self.in_dim = in_dim
        self.out_dim = in_dim*2 + in_dim*2*dim_z
        if use_pos_encoding:
            in_dim = 2 + 2 * 2 * num_frequencies  # sin/cos for each x1 and x2
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.SiLU(), 
                                 nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(),
                                 nn.Linear(64, self.out_dim))
        self.f2_param = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))

    def forward(self, x): # x shape(Nt, L, 2)
        Nt, L = x.shape[:2]
        if self.use_pos_encoding:
            x = self.positional_encoding(x.view(-1,2)) # (Nt * L, in_dim)
            x = x.view(Nt, L, -1)
        out = self.net(x) # (Nt, L, out_dim)
        f1 = out[:, :, :self.in_dim*2].reshape(Nt, *self.f1_size) # (Nt, L*4, 1)
        g1 = out[:, :, self.in_dim*2:].reshape(Nt, *self.g1_size) # (Nt, L*4, dim_z)
        g1 = nnF.softmax(g1 / 1, dim=-1) # sharp attention 
        f2 = self.f2_param.repeat(Nt, 1, 1)
        g2 = self.g2_param.repeat(Nt, 1, 1)
        return [f1, g1, f2, g2]

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

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u1, z): # u1: (Nt, L, 4), z: (Nt, dim_z)
        # Matrix-form Computation
        x, y = unit2xy(u1)
        pos = torch.stack([x, y], dim=-1) # (Nt, L, 2)
        f1, g1, f2, g2 = self.cgn(pos) # f1: (Nt, L*4, 1), g1: (Nt, L*4, dim_z), f2: (Nt, dim_z, 1), g2: (Nt, dim_z, dim_z)
        z = z.unsqueeze(-1) # (Nt, dim_z, 1)
        u1_pred = f1 + g1@z # (Nt, L*4, 1)
        z_pred = f2 + g2@z  # (Nt, dim_z, 1)
        return u1_pred.view(*u1.shape), z_pred.squeeze(-1)


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = 128*4
dim_u2 = 64*64*2
z_h, z_w = (16, 16)
dim_z = z_h*z_w*2

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 500
train_batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_forecast_u1_history = []
train_loss_forecast_u2_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
train_loss_physics_history = []

autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
field2point = Field2point(K=64).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
for ep in range(1, epochs+1):
    cgkn.train()
    start_time = time.time()

    train_loss_forecast_u1 = 0.
    train_loss_forecast_u2 = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    train_loss_physics = 0.
    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # AutoEncoder
        z_initial = cgkn.autoencoder.encoder(u2_initial)
        u2_initial_ae = cgkn.autoencoder.decoder(z_initial)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)

        #  State Forecast
        z_initial_flat = z_initial.view(-1, dim_z)
        u1_pred, z_flat_pred = cgkn(u1_initial, z_initial_flat)
        z_pred = z_flat_pred.view(-1, 2, z_h, z_w)
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_pred)

        # x_next, y_next = unit2xy(u1_next)
        # # x_pred, y_pred = unit2xy(u1_pred)
        # psi_next = u2_next[..., 0]
        # psi_pred = u2_pred[..., 0]
        # psi_xy_next = field2point(x_next, y_next, psi_next) # shape (B, L)
        # psi_xy_pred = field2point(x_next, y_next, psi_pred) # shape (B, L)
        # # psi_xy_next1 = field2point(x_pred, y_pred, psi_next) # shape (B, L)
        # # psi_xy_pred1 = field2point(x_pred, y_pred, psi_pred) # shape (B, L)
        # loss_physics = nnF.mse_loss(psi_xy_next, psi_xy_pred) #+ nnF.mse_loss(psi_xy_next1, psi_xy_pred1)

        loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z #+ loss_physics

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
        # train_loss_physics += loss_physics.item()
    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_physics /= train_num_batches
    train_loss_forecast_u1_history.append(train_loss_forecast_u1)
    train_loss_forecast_u2_history.append(train_loss_forecast_u2)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)
    # train_loss_physics_history.append(train_loss_physics)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast_u1, 4), round(train_loss_forecast_u2, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4),
          # " loss phy:", round(train_loss_physics, 4)
          )

torch.save(cgkn, r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_stage1.pt")
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_train_loss_forecast_u1_history_stage1.npy", train_loss_forecast_u1_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_train_loss_forecast_u2_history_stage1.npy", train_loss_forecast_u2_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_train_loss_ae_history_stage1.npy", train_loss_ae_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)
# np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_train_loss_physics_history_stage1.npy", train_loss_physics_history)

# """
# cgkn = torch.load(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_stage1.pt").to(device)

# # Model Diagnosis in Physical Space
# train_u1 = train_u1.to(device)
# train_u2 = train_u2.to(device)
# with torch.no_grad():
#    train_z_concat = cgkn.autoencoder.encoder(train_u2).view(Ntrain, dim_z)
#    train_u_extended = torch.cat([train_u1, train_z_concat], dim=-1)
#    train_u_extended_pred = cgkn(train_u_extended)
#    train_u1_pred = train_u_extended_pred[:, :dim_u1]
#    train_z_pred = train_u_extended_pred[:, dim_u1:].view(Ntrain, int(dim_z**0.5), int(dim_z**0.5))
#    train_u2_pred = cgkn.autoencoder.decoder(train_z_pred)
# MSE1 = nnF.mse_loss(train_u1[1:], train_u1_pred[:-1])
# print("train MSE1:", MSE1.item())
# MSE2 = nnF.mse_loss(train_u2[1:], train_u2_pred[:-1])
# print("train MSE2:", MSE2.item())
# train_u1 = train_u1.to("cpu")
# train_u2 = train_u2.to("cpu")
# del train_u1_pred, train_u2_pred, train_z_pred, train_z_concat, train_u_extended, train_u_extended_pred
# torch.cuda.empty_cache()

# CGKN for One-Step Prediction
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
with torch.no_grad():
    test_z_flat = cgkn.autoencoder.encoder(test_u2).view(Ntest, dim_z)
    test_u1_pred, test_z_flat_pred = cgkn(test_u1, test_z_flat)
    test_z_pred = test_z_flat_pred.view(Ntest, 2, z_h, z_w)
    test_u2_pred = cgkn.autoencoder.decoder(test_z_pred)
MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")

np.save(r"../data/CGKN_64x64_sigmaxy01_xy_unit_OneStepPrediction_stage1.npy", test_u1_pred.to("cpu"))
np.save(r"../data/CGKN_64x64_sigmaxy01_psi_OneStepPrediction_stage1.npy", test_u2_pred.to("cpu"))


#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################

# train_u1 = train_u1.to(device)
# train_u2 = train_u2.to(device)
# with torch.no_grad():
#     train_z_concat = cgkn.autoencoder.encoder(train_u2).view(Ntrain, dim_z)
#     train_u_extended = torch.cat([train_u1, train_z_concat], dim=-1)
#     train_u_extended_pred = cgkn(train_u_extended)
#     train_u1_pred = train_u_extended_pred[:, :dim_u1]
# sigma_hat = torch.zeros(dim_u1 + dim_z)
# sigma_hat[:dim_u1] = torch.sqrt(torch.mean((train_u1[1:] - train_u1_pred[:-1])**2, dim=0)).to("cpu")
# sigma_hat[dim_u1:] = 0.5 # sigma2 is set manually
# train_u1 = train_u1.to("cpu")
# train_u2 = train_u2.to("cpu")
# del train_u1_pred, train_u_extended_pred, train_u_extended, train_z_concat
# torch.cuda.empty_cache()

batch_size = 100
# Move data to device first if it fits, or load per batch if necessary
train_u1 = train_u1.to(device)
train_u2 = train_u2.to(device)
train_u1_preds = []
with torch.no_grad():
    for i in range(0, Ntrain, batch_size):
        batch_u1 = train_u1[i:i+batch_size]
        batch_u2 = train_u2[i:i+batch_size]
        batch_z_flat = cgkn.autoencoder.encoder(batch_u2).view(batch_u2.size(0), dim_z)
        batch_u1_pred, _ = cgkn(batch_u1, batch_z_flat)
        train_u1_preds.append(batch_u1_pred)
    train_u1_pred = torch.cat(train_u1_preds, dim=0)
sigma_hat = torch.zeros(dim_u1 + dim_z)
sigma_hat[:dim_u1] = torch.sqrt(torch.mean((train_u1[1:] - train_u1_pred[:-1])**2, dim=0)).view(-1).to("cpu")
sigma_hat[dim_u1:] = 0.1  # sigma2 manually set
train_u1 = train_u1.to("cpu")
train_u2 = train_u2.to("cpu")
del train_u1_pred, train_u1_preds, batch_u1_pred, batch_z_flat
torch.cuda.empty_cache()

def CGFilter(cgkn, sigma, u1, mu0, R0):
    # u1: (Nt, L, 4, 1)
    # mu0: (dim_z, 1)
    # R0: (dim_z, dim_z)
    device = u1.device
    Nt = u1.shape[0]
    u1_flat = u1.view(Nt, -1, 1)
    dim_u1 = u1_flat.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    mu_pred = torch.zeros((Nt, dim_z, 1)).to(device)
    R_pred = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_pred[0] = mu0
    R_pred[0] = R0
    for n in range(1, Nt):
        x, y = unit2xy(u1[n-1].permute(2,0,1))
        pos = torch.stack([x, y], dim=-1) # (1, L, 2)
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(pos)]

        K0 = torch.linalg.solve(s1 @ s1.T + g1 @ R0 @ g1.T, g1 @ R0 @ g2.T).T
        mu1 = f2 + g2@mu0 + K0@(u1_flat[n]-f1-g1@mu0)
        R1 = g2@R0@g2.T + s2@s2.T - K0@g1@R0@g2.T
        R1 = 0.5 * (R1 + R1.T)

        # mu1 = f2 + g2@mu0 + K@(u1_flat[n]-f1-g1@mu0)
        # R1 = g2@R0@g2.T + s2@s2.T - K@g1@R0@g2.T
        mu_pred[n] = mu1
        R_pred[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_pred, R_pred)

def CGFilter_batch(cgkn, sigma, u1, mu0, R0):
    # u1:  (B, Nt, L, 4, 1)
    # mu0: (B, dim_z, 1)
    # R0:  (B, dim_z, dim_z)
    device = u1.device
    B, Nt = u1.shape[:2]
    u1_flat = u1.view(B, Nt, -1, 1)  # (B, Nt, dim_u1, 1)
    dim_u1 = u1_flat.shape[2]
    dim_z = mu0.shape[1]
    s1 = torch.diag(sigma[:dim_u1]).to(device)       # (dim_u1, dim_u1)
    s2 = torch.diag(sigma[dim_u1:]).to(device)       # (dim_z, dim_z)
    s1_cov = s1 @ s1.T  # (dim_u1, dim_u1)
    s2_cov = s2 @ s2.T  # (dim_z, dim_z)
    mu_pred = torch.zeros((B, Nt, dim_z, 1), device=device)
    R_pred = torch.zeros((B, Nt, dim_z, dim_z), device=device)
    mu_pred[:, 0] = mu0
    R_pred[:, 0] = R0
    for n in range(1, Nt):
        x, y = unit2xy(u1[:, n - 1].squeeze(-1))  # u1[:, n-1]: (B, L, 4, 1)
        pos = torch.stack([x, y], dim=-1)  # (B, L, 2)
        f1, g1, f2, g2 = cgkn.cgn(pos)  # shapes: (B, dim_u1, 1), (B, dim_u1, dim_z), ...

        # Kalman gain: solve per batch
        S = s1_cov + torch.bmm(g1, torch.bmm(R0, g1.transpose(1, 2)))      # (B, dim_u1, dim_u1)
        Cross = torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2)))           # (B, dim_u1, dim_z)
        K = torch.linalg.solve(S, Cross).transpose(1, 2)                   # (B, dim_z, dim_u1)
        innov = u1_flat[:, n] - f1 - torch.bmm(g1, mu0)                    # (B, dim_u1, 1)
        mu1 = f2 + torch.bmm(g2, mu0) + torch.bmm(K, innov)                # (B, dim_z, 1)
        R1 = torch.bmm(g2, torch.bmm(R0, g2.transpose(1, 2))) + s2_cov \
             - torch.bmm(K, torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2))))  # (B, dim_z, dim_z)
        R1 = 0.5 * (R1 + R1.transpose(1, 2))  # Ensure symmetry

        mu_pred[:, n] = mu1
        R_pred[:, n] = R1
        mu0 = mu1
        R0 = R1
    return mu_pred, R_pred

########################################################
################# Train cgkn (Stage2)  #################
########################################################
torch.manual_seed(2)
np.random.seed(2)

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = 2
long_steps = 100
cut_point = 20

epochs = 500
train_batch_size = 200
train_batch_size_da = 10
train_num_batches = int(Ntrain / train_batch_size)
Niters = epochs * train_num_batches
train_loss_forecast_u1_history = []
train_loss_forecast_u2_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
train_loss_physics_history = []
# # Re-initialize Model
# autoencoder = AutoEncoder().to(device)
# cgn = CGN(dim_u1, dim_z).to(device)
# cgkn = CGKN(autoencoder, cgn).to(device)
# field2point = Field2point(K=128).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
cgkn.train()
for itr in range(1, Niters+1):
    # print(itr)
    start_time = time.time()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u1_short = torch.stack([train_u1[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (short_steps, Nt, L, 2)
    u2_short = torch.stack([train_u2[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (short_steps, Nt, Nx, Nx, 2)

    # AutoEncoder
    z_short = cgkn.autoencoder.encoder(u2_short.view(-1, *u2_short.shape[2:])) # (short_steps*Nt, 2, 16, 16)
    u2_ae_short = cgkn.autoencoder.decoder(z_short).view(*u2_short.shape)      # (short_steps, Nt, Nx, Nx, 2)
    loss_ae = nnF.mse_loss(u2_short, u2_ae_short)

    # State Prediction
    z_short = z_short.view(short_steps, train_batch_size, *z_short.shape[1:]) # (short_steps, Nt, 2, 16, 16)
    z_flat_short = z_short.reshape(short_steps, train_batch_size, dim_z)      # (short_steps, Nt, dim_z)
    u1_short_pred = [u1_short[0]]
    z_flat_short_pred = [z_flat_short[0]]
    for n in range(1, short_steps):
        u1_short_pred_n, z_short_pred_n = cgkn(u1_short_pred[-1], z_flat_short_pred[-1])
        u1_short_pred.append(u1_short_pred_n)
        z_flat_short_pred.append(z_short_pred_n)
    u1_short_pred = torch.stack(u1_short_pred, dim=0)
    z_flat_short_pred = torch.stack(z_flat_short_pred, dim=0)

    loss_forecast_z = nnF.mse_loss(z_flat_short[1:], z_flat_short_pred[1:])

    z_short_pred = z_flat_short_pred.reshape(short_steps, train_batch_size, 2, z_h, z_w)
    u2_short_pred = cgkn.autoencoder.decoder(z_short_pred.view(-1, 2, z_h, z_w)).view(*u2_short.shape)
    loss_forecast_u1 = nnF.mse_loss(u1_short[1:], u1_short_pred[1:])
    loss_forecast_u2 = nnF.mse_loss(u2_short[1:], u2_short_pred[1:])

    # x_next, y_next = unit2xy(u1_short[1])
    # # x_pred, y_pred = unit2xy(u1_short_pred[1])
    # psi_next = u2_short[1][..., 0]
    # psi_pred = u2_short_pred[1][..., 0]
    # psi_xy_next = field2point(x_next, y_next, psi_next) # shape (B, L)
    # psi_xy_pred = field2point(x_next, y_next, psi_pred) # shape (B, L)
    # # psi_xy_next1 = field2point(x_pred, y_pred, psi_next) # shape (B, L)
    # # psi_xy_pred1 = field2point(x_pred, y_pred, psi_pred) # shape (B, L)
    # loss_physics = nnF.mse_loss(psi_xy_next, psi_xy_pred) #+ nnF.mse_loss(psi_xy_next1, psi_xy_pred1)

    # # DA
    # head_idx_long = torch.from_numpy(np.random.choice(Ntrain-long_steps+1, size=1, replace=False))
    # u1_long = train_u1[head_idx_long:head_idx_long + long_steps].to(device)
    # u2_long = train_u2[head_idx_long:head_idx_long + long_steps].to(device)
    # mu_z_flat_pred_long = CGFilter(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
    # mu_z_pred_long = mu_z_flat_pred_long.reshape(-1,  2, z_h, z_w)
    # mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
    # loss_da = nnF.mse_loss(u2_long[cut_point:], mu_pred_long)

    # Batch DA
    head_idx_long = torch.from_numpy(np.random.choice(Ntrain - long_steps + 1, size=train_batch_size_da, replace=False))
    u1_long = torch.stack([train_u1[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, L, 4)
    u2_long = torch.stack([train_u2[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, H, W, 2)
    mu0 = torch.zeros(train_batch_size_da, dim_z, 1, device=device)
    R0 = 0.1 * torch.eye(dim_z, device=device).expand(train_batch_size_da, dim_z, dim_z)
    mu_z_flat_pred_long, _ = CGFilter_batch(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0, R0)  # (B, Nt, dim_z, 1)
    mu_z_pred_long = mu_z_flat_pred_long[:, cut_point:].squeeze(-1).reshape(-1, 2, z_h, z_w)  # (B*(Nt-cut), 2, z_h, z_w)
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long)  # (B*(Nt-cut), H, W, 2)
    mu_pred_long = mu_pred_long.view(train_batch_size_da, long_steps - cut_point, *mu_pred_long.shape[1:])  # (B, Nt-cut, ...)
    u2_long_target = u2_long[:, cut_point:]  # (B, Nt-cut, H, W, 2)
    loss_da = nnF.mse_loss(mu_pred_long, u2_long_target)

    loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z + loss_da #+ loss_physics

    if torch.isnan(loss_total):
        print(itr, "nan")
        continue

    # print(mu_pred_long.requires_grad)  # Should be True
    # print(mu_z_pred_long.requires_grad)
    # print(mu_z_concat_pred_long.requires_grad)

    # # Visualize the computational graph of loss_da
    # if itr == 1:
    #     dot = make_dot(loss_da, params=dict(cgkn.named_parameters()))
    #     dot.format = "pdf"
    #     dot.directory = "graph"
    #     dot.render("cgkn_loss_da_graph")

    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    train_loss_forecast_u1_history.append(loss_forecast_u1.item())
    train_loss_forecast_u2_history.append(loss_forecast_u2.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae_history.append(loss_ae.item())
    train_loss_forecast_z_history.append(loss_forecast_z.item())
    # train_loss_physics_history.append(loss_physics.item())

    end_time = time.time()
    print("itr", itr,
          " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast_u1.item(), 4), round(loss_forecast_u2.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae:", round(loss_ae.item(),4),
          " loss fore z:", round(loss_forecast_z.item(), 4),
          # " loss phy:", round(loss_physics.item(), 4),
          )
 
# """

torch.save(cgkn, r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_stage2.pt")
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_train_loss_forecast_u1_history_stage2.npy", train_loss_forecast_u1_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_train_loss_forecast_u2_history_stage2.npy", train_loss_forecast_u2_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_train_loss_ae_history_stage2.npy", train_loss_ae_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)
np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_train_loss_da_history_stage2.npy", train_loss_da_history)
# np.save(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_physics_dimz256_batchda_train_loss_physics_history_stage2.npy", train_loss_physics_history)


# cgkn = torch.load(r"../model/QG(Noisy)_64x64_sigmaxy01_CGKN_dimz256_batchda_stage2.pt").to(device)


############################################
################ Test cgkn #################
############################################

# CGKN for One-Step Prediction
batch_size = 100
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
test_u1_preds = []
test_u2_preds = []
cgkn.eval()
with torch.no_grad():
    for i in range(0, Ntest, batch_size):
        test_u1_batch = test_u1[i:i+batch_size]
        test_u2_batch = test_u2[i:i+batch_size]
        test_z_flat_batch = cgkn.autoencoder.encoder(test_u2_batch).view(batch_size, dim_z)
        test_u1_pred_batch, test_z_flat_pred_batch = cgkn(test_u1_batch, test_z_flat_batch)
        test_z_pred_batch = test_z_flat_pred_batch.view(batch_size, 2, z_h, z_w)
        test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_pred_batch)
        test_u1_preds.append(test_u1_pred_batch)
        test_u2_preds.append(test_u2_pred_batch)
    test_u1_pred = torch.cat(test_u1_preds, dim=0)
    test_u2_pred = torch.cat(test_u2_preds, dim=0)
MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")

np.save(r"../data/CGKN_64x64_batchda_sigmaxy01_xy_unit_OneStepPrediction.npy", test_u1_pred.to("cpu"))
np.save(r"../data/CGKN_64x64_batchda_sigmaxy01_psi_OneStepPrediction.npy", test_u2_pred.to("cpu"))


# CGKN for Data Assimilation
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
cgkn.eval()
with torch.no_grad():
    test_mu_z_flat_pred = CGFilter(cgkn, sigma_hat.to(device), test_u1.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
    test_mu_z_pred = test_mu_z_flat_pred.reshape(-1, 2, z_h, z_w)
    test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred)
MSE2_DA = nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:])
print("MSE2_DA:", MSE2_DA.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/CGKN_64x64_batchda_sigmaxy01_psi_DA.npy", test_mu_pred.to("cpu"))

# CGKN: Number of Parameters
cgn_params = parameters_to_vector(cgkn.cgn.parameters()).numel()
encoder_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
decoder_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
total_params = cgn_params + encoder_params + decoder_params
print(f'cgn #parameters:      {cgn_params:,}')
print(f'encoder #parameters:  {encoder_params:,}')
print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')

# # CGKN for Data Assimilation
# train_u1 = u1[Ntrain-2000:Ntrain]
# train_u2 = u2[Ntrain-2000:Ntrain]
# test_u1 = train_u1.to(device)
# test_u2 = train_u2.to(device)
# with torch.no_grad():
#     test_mu_z_flat_pred = CGFilter(cgkn, sigma_hat.to(device), test_u1.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
#     test_mu_z_pred = test_mu_z_flat_pred.reshape(-1, 2, z_h, z_w)
#     test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred)
# MSE2_DA = nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:])
# print("MSE2_DA:", MSE2_DA.item())
# test_u1 = test_u1.to("cpu")
# test_u2 = test_u2.to("cpu")
# np.save(r"../data/CGKN_64x64_physics_sigmaxy01_psi_DA_trainingdata.npy", test_mu_pred.to("cpu"))
