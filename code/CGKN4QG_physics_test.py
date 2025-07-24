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

QG_Data = np.load(r"../data/qg_data_sigmaxy01_t1000_subsampled.npz")

pos = QG_Data["xy_obs"]
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1)
psi = QG_Data["psi_noisy"]

# u1 = torch.tensor(pos_unit, dtype=torch.float).view(pos.shape[0], -1)
u1 = torch.tensor(pos_unit, dtype=torch.float) # shape (Nt, L, 4), keep tracers parallel
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 128, 128, 2)

# Train / Test
Ntrain = 20000
Ntest = 5000

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
          " loss fore z:", round(train_loss_forecast_z, 4),)
          # " loss phy:", round(train_loss_physics, 4))

# CGKN: Number of Parameters
cgn_params = parameters_to_vector(cgkn.cgn.parameters()).numel()
encoder_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
decoder_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
total_params = cgn_params + encoder_params + decoder_params
print(f'cgn #parameters:      {cgn_params:,}')
print(f'encoder #parameters:  {encoder_params:,}')
print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')
