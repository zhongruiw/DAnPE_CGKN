import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import time

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
train_u2_original = u2[:Ntrain]
test_u2_original = u2[Ntrain:Ntrain+Ntest]

# # Normalization
# class Normalizer:
#     def __init__(self, x, eps=1e-9):
#         # x is in the shape tensor (N, y, x, 2)
#         self.mean = torch.mean(x, dim=0)
#         self.std = torch.std(x, dim=0)
#         self.eps = eps

#     def encode(self, x):
#         return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

#     def decode(self, x):
#         return x*(self.std.to(x.device) + self.eps) + self.mean.to(x.device)
# normalizer = Normalizer(train_u2_original)
# train_u = normalizer.encode(train_u2_original)
# test_u = normalizer.encode(test_u2_original)

train_u = train_u2_original
test_u = test_u2_original


################################################################
################# CNN: Unit Solution Operator  #################
################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class UNetQG(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(2, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256)

        # Decoder
        self.dec3 = ConvBlock(256 + 256, 128)
        self.dec2 = ConvBlock(128 + 128, 64)
        self.dec1 = ConvBlock(64 + 64, 64)

        # Final conv
        self.final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # Input shape: (B, H, W, 2)
        x = x.permute(0, 3, 1, 2)  # → (B, 2, H, W)

        # Encoder
        e1 = self.enc1(x)         # → (B, 64, H, W)
        e2 = self.enc2(F.avg_pool2d(e1, 2))  # → (B, 128, H/2, W/2)
        e3 = self.enc3(F.avg_pool2d(e2, 2))  # → (B, 256, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(F.avg_pool2d(e3, 2))  # → (B, 256, H/8, W/8)

        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], dim=1))

        out = self.final(d1)  # → (B, 2, H, W)
        return out.permute(0, 2, 3, 1)  # → (B, H, W, 2)

class SolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(32, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 32, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(32, 2, 3, 1, 1))
    def forward(self, x):
        # x shape(t, y, x, 2)
        x = x.permute(0, 3, 1, 2)
        out = self.seq(x)
        out = out.permute(0, 2, 3, 1)
        return out

##############################################
################# Train CNN  #################
##############################################
# """
epochs = 500
batch_size = 200
train_tensor = torch.utils.data.TensorDataset(train_u[:-1], train_u[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_forecast_history = []

solnet = SolNet().to(device)
optimizer = torch.optim.Adam(solnet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for ep in range(1, epochs+1):
    solnet.train()
    start_time = time.time()

    train_loss_forecast = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)
        u_pred = solnet(u)
        loss_forecast = nnF.mse_loss(u_next, u_pred)
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        optimizer.zero_grad()
        loss_forecast.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast += loss_forecast.item()
    train_loss_forecast /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4))

torch.save(solnet, r"../model/QG_64x64_CNN_unnormalized.pt")
np.save(r"../model/QG_64x64_CNN_unnormalized_train_loss_forecast_history.npy", train_loss_forecast_history)

# CGKN: Number of Parameters
cnn_params = parameters_to_vector(solnet.parameters()).numel()
print(f'cnn #parameters:      {cnn_params:,}')
# """
############################################
################# Test cnn #################
############################################
# solnet = torch.load(r"../model/QG_64x64_CNN_unnormalized.pt").to(device)

# One-step Prediction
test_batch_size = 200
test_tensor = torch.utils.data.TensorDataset(test_u)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u_pred = torch.zeros_like(test_u)
si = 0
for u in test_loader:
    solnet.eval()
    test_u_batch = u[0].to(device)
    with torch.no_grad():
        test_u_pred[si:si+test_batch_size] = solnet(test_u_batch)
    si += test_batch_size
# print("MSE (normalized):", nnF.mse_loss(test_u[1:], test_u_pred[:-1]).item())
# test_u_original_pred = normalizer.decode(test_u_pred)
test_u_original_pred = test_u_pred
print("MSE (original):",nnF.mse_loss(test_u2_original[1:], test_u_original_pred[:-1]).item())
np.save(r"../data/CNN_psi_64x64_OneStepPrediction.npy", test_u_original_pred.to("cpu"))

# # Multi-step Prediction
# test_short_steps = 10
# mask = torch.ones(Ntest, dtype=torch.bool)
# mask[::test_short_steps] = False
# test_u_initial = test_u[::test_short_steps]
# test_batch_size = 100
# test_tensor = torch.utils.data.TensorDataset(test_u_initial)
# test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
# test_u_shortPred = torch.zeros(test_short_steps*test_u_initial.shape[0], 64, 64)
# si = 0
# for u in test_loader:
#     test_u_initial_batch = u[0].to(device)
#     test_u_shortPred_batch = torch.zeros(test_short_steps, *test_u_initial_batch.shape).to(device)
#     test_u_shortPred_batch[0] = test_u_initial_batch
#     with torch.no_grad():
#         for n in range(test_short_steps-1):
#             test_u_shortPred_batch[n+1] = solnet(test_u_shortPred_batch[n])
#     test_u_shortPred_batch = test_u_shortPred_batch.permute(1, 0, 2, 3).reshape(-1, 64, 64)
#     test_u_shortPred[si:si+test_u_shortPred_batch.shape[0]] = test_u_shortPred_batch
#     si = si+test_u_shortPred_batch.shape[0]
# test_u_shortPred = test_u_shortPred[:Ntest]
# print(nnF.mse_loss(test_u[mask], test_u_shortPred[mask]).item())
# test_u_original_shortPred = normalizer.decode(test_u_shortPred)
# print(nnF.mse_loss(test_u_original[mask], test_u_original_shortPred[mask]).item())

# # Multi-step Prediction (Data start from 950)
# si = 15000
# steps = 11
# u_shortPred = torch.zeros(steps, 1, 64, 64).to(device)
# u_shortPred[0] = test_u[[si]].to(device)
# with torch.no_grad():
#     for n in range(steps-1):
#         u_shortPred[n+1] = solnet(u_shortPred[n])
# u_original_shortPred = normalizer.decode(u_shortPred.squeeze(1).cpu())
# # np.save(path_abs + r"/Data/NSE(Noisy)_CNN_950initial_10steps.npy", u_original_shortPred)
