import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector
import time
from torchviz import make_dot

device = "cuda:0"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_sigmaxy01_t1000_subsampled.npz")
psi = QG_Data["psi_noisy"]
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 128, 128, 2)

# Train / Test
Ntrain = 20000
Ntest = 5000
train_u2 = u2[:Ntrain]
test_u2 = u2[Ntrain:Ntrain+Ntest]

############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes  # Number of Fourier modes to keep
#         self.scale = 1 / (in_channels * out_channels)
#         self.weights = nn.Parameter(
#             self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
#         )

#     def compl_mul2d(self, input, weights):
#         # (B, in_c, H, W), (in_c, out_c, H, W)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_ft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)

#         out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(
#             x_ft[:, :, :self.modes, :self.modes], self.weights
#         )

#         x_out = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
#         return x_out

# class FNOBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, modes, width):
#         super().__init__()
#         self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.activation = nn.GELU()

#     def forward(self, x):
#         return self.activation(self.spectral_conv(x) + self.pointwise(x))

# class FNOEncoder(nn.Module):
#     def __init__(self, in_channels=2, width=32, modes=16, depth=4):
#         super().__init__()
#         self.input_proj = nn.Conv2d(in_channels, width, 1)
#         self.blocks = nn.Sequential(*[
#             FNOBlock(width, width, modes, width) for _ in range(depth)
#         ])
#         self.output_proj = nn.Conv2d(width, width, 1)

#     def forward(self, x):
#         x = self.input_proj(x)
#         x = self.blocks(x)
#         return self.output_proj(x)

# class FNODecoder(nn.Module):
#     def __init__(self, out_channels=2, width=32, modes=16, depth=4):
#         super().__init__()
#         self.blocks = nn.Sequential(*[
#             FNOBlock(width, width, modes, width) for _ in range(depth)
#         ])
#         self.output_proj = nn.Conv2d(width, out_channels, 1)

#     def forward(self, x):
#         x = self.blocks(x)
#         return self.output_proj(x)

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
        # self.enc1 = nn.Sequential(
        #     CircularConv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(32, 1, kernel_size=3, stride=1, padding=1)
        # )
        # self.enc2 = nn.Sequential(
        #     CircularConv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(32, 1, kernel_size=3, stride=1, padding=1)
        # )
        # self.combine = nn.Sequential(
        #     CircularConv2d(2, 8, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConv2d(8, 2, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, u):                           # u:(B, H, W, 2)
        u = u.permute(0, 3, 1, 2)                   # → (B, 2, H, W)
        out = self.enc(u)                           # → (B, 2, d1, d2)
        # e1 = self.enc1(u[:, 0:1])                   # → (B, 1, d1, d2)
        # e2 = self.enc2(u[:, 1:2])                   # → (B, 1, d1, d2)
        # e_cat = torch.cat([e1, e2], dim=1)          # → (B, 2, d1, d2)
        # out = self.combine(e_cat)                   # → (B, 2, d1, d2)
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
        # self.expand = nn.Sequential(
        #     CircularConvTranspose2d(2, 8, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1) # Split into 2 channels
        # )
        # self.dec1 = nn.Sequential(
        #     CircularConvTranspose2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        # )
        # self.dec2 = nn.Sequential(
        #     CircularConvTranspose2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #     CircularConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, z):                  # z: (B, 2, d1, d2)
        # z = z.unsqueeze(1)                 # (B, 1, d1, d2)
        u = self.dec(z)                    # (B, 2, 64, 64)
        # e_cat = self.expand(z)             # (B, 2, d1, d2)
        # e1 = e_cat[:, 0:1, :, :]           # (B, 1, d1, d2)
        # e2 = e_cat[:, 1:2, :, :]           # (B, 1, d1, d2)
        # u1 = self.dec1(e1)                 # (B, 1, 64, 64)
        # u2 = self.dec2(e2)                 # (B, 1, 64, 64)
        # u = torch.cat([u1, u2], dim=1)     # (B, 2, 64, 64)
        return u.permute(0, 2, 3, 1)       # (B, 64, 64, 2)
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.f2_param = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))

    def forward(self, batch_size):
        f2 = self.f2_param.repeat(batch_size, 1, 1)
        g2 = self.g2_param.repeat(batch_size, 1, 1)
        return [f2, g2]

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, z):
        # Matrix-form Computation
        f2, g2 = self.cgn(z.shape[0])
        z = z.unsqueeze(-1)
        z_pred = f2 + g2@z
        return z_pred.squeeze(-1)


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u2 = 64*64*2
z_h, z_w = (16, 16)
dim_z = z_h*z_w*2

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 500
train_batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u2[:-1], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_forecast_u2_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    cgkn.train()
    start_time = time.time()

    train_loss_forecast_u2 = 0.    
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u2_initial, u2_next in train_loader:
        u2_initial, u2_next = u2_initial.to(device), u2_next.to(device)

        # AutoEncoder
        z_initial = cgkn.autoencoder.encoder(u2_initial)
        u2_initial_ae = cgkn.autoencoder.decoder(z_initial)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)

        #  State Forecast
        z_initial_flat = z_initial.view(-1, dim_z)
        z_flat_pred = cgkn(z_initial_flat)
        z_pred = z_flat_pred.view(-1, 2, z_h, z_w)
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_pred)

        loss_total = loss_forecast_u2 + loss_ae + loss_forecast_z

        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u2 += loss_forecast_u2.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
    train_loss_forecast_u2 /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_forecast_u2_history.append(train_loss_forecast_u2)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast_u2, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))

torch.save(cgkn, r"../model/QG_64x64_EncoderCG.pt")
np.save(r"../model/QG_64x64_EncoderCG_train_loss_forecast_u2_history_stage1.npy", train_loss_forecast_u2_history)
np.save(r"../model/QG_64x64_EncoderCG_train_loss_ae_history_stage1.npy", train_loss_ae_history)
np.save(r"../model/QG_64x64_EncoderCG_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

"""
cgkn = torch.load(r"../model/QG_64x64_EncoderCG.pt").to(device)

# CGKN for One-Step Prediction
test_u2 = test_u2.to(device)
cgkn.eval()
with torch.no_grad():
    test_z_flat = cgkn.autoencoder.encoder(test_u2).view(Ntest, dim_z)
    test_z_flat_pred = cgkn(test_z_flat)
    test_z_pred = test_z_flat_pred.view(Ntest, 2, z_h, z_w)
    test_u2_pred = cgkn.autoencoder.decoder(test_z_pred)
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u2 = test_u2.to("cpu")

np.save(r"../data/EncoderCG_psi_64x64_OneStepPrediction.npy", test_u2_pred.to("cpu"))

# CGKN: Number of Parameters
cgn_params = parameters_to_vector(cgkn.cgn.parameters()).numel()
encoder_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
decoder_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
total_params = cgn_params + encoder_params + decoder_params
print(f'cgn #parameters:      {cgn_params:,}')
print(f'encoder #parameters:  {encoder_params:,}')
print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')