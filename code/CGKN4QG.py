import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load("../data/qg_data.npz")
pos = QG_Data["xy_obs"]
psi = QG_Data["psi_noisy"]

u1 = torch.tensor(pos, dtype=torch.float).reshape(pos.shape[0], -1)
u2 = torch.tensor(psi, dtype=torch.float)


# Train / Test
Ntrain = 1600
Ntest = 400

train_u1 = u1[:Ntrain]
train_u2 = u2[:Ntrain]
test_u1 = u1[Ntrain:Ntrain+Ntest]
test_u2 = u2[Ntrain:Ntrain+Ntest]


############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        # x shape(t, y, x, 2)
        x = x.permute(0, 3, 1, 2)
        out = self.seq(x)
        out = out.squeeze(1)
        return out
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.ConvTranspose2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        # x shape(t, d1, d2)
        x = x.unsqueeze(1)
        x = self.seq(x)
        out = x.permute(0, 2, 3, 1)
        return out
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.f1_size = (dim_u1, 1)
        self.g1_size = (dim_u1, dim_z)
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.input_size = 2
        self.output_size = self.input_size + self.input_size*dim_z
        self.net = nn.Sequential(nn.Linear(self.input_size, 8), nn.SiLU(),
                                 nn.Linear(8, 16), nn.SiLU(),
                                 nn.Linear(16, 32), nn.SiLU(),
                                 nn.Linear(32, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(16, self.output_size))
        self.f2_param = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))

    def forward(self, x):
        # x shape (Nt, 128*2)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.input_size)
        out = self.net(x)
        f1 = out[:, :, :self.input_size].reshape(batch_size, *self.f1_size)
        g1 = out[:, :, self.input_size:].reshape(batch_size, *self.g1_size)
        f2 = self.f2_param.repeat(batch_size, 1, 1)
        g2 = self.g2_param.repeat(batch_size, 1, 1)
        return [f1, g1, f2, g2]

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u_extended):
        # Matrix Form Computation
        dim_u1 = self.cgn.dim_u1
        u1 = u_extended[:, :dim_u1]
        z = u_extended[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn(u1)
        z = z.unsqueeze(-1)
        u1_pred = f1 + g1@z
        z_pred = f2 + g2@z
        u_extended_next = torch.cat([u1_pred.squeeze(-1), z_pred.squeeze(-1)], dim=-1)
        return u_extended_next



########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = 128*2
dim_u2 = 128*128*2
dim_z = 16*16

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 1000
batch_size = 100
train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_forecast = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # AutoEncoder
        z_initial = cgkn.autoencoder.encoder(u2_initial)
        u2_initial_ae = cgkn.autoencoder.decoder(z_initial)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)


        #  State Forecast
        z_initial_concat = z_initial.view(-1, dim_z)
        u_extended = torch.cat([u1_initial, z_initial_concat], dim=-1)
        u_extended_pred = cgkn(u_extended)
        u1_next_pred = u_extended_pred[:, :dim_u1]
        z_next_concat_pred = u_extended_pred[:, dim_u1:] # (N, 256)
        z_next_pred = z_next_concat_pred.view(-1, int(dim_z**0.5), int(dim_z**0.5))
        u2_next_pred = cgkn.autoencoder.decoder(z_next_pred)
        u1_err = u1_next - u1_next_pred
        u1_err = (u1_err+torch.pi) % (2*torch.pi) - torch.pi
        mse_u1 = torch.mean(u1_err**2)
        mse_u2 = nnF.mse_loss(u2_next, u2_next_pred)
        loss_forecast = mse_u1 + mse_u2

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_next_pred)

        loss_total = loss_forecast + loss_ae + loss_forecast_z
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_forecast += loss_forecast.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
    train_loss_forecast /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))
"""

# torch.save(cgkn, r"../model/QG(Noisy)_CGKN__dimz256_stage1.pt")
# np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_forecast_history_stage1.npy", train_loss_forecast_history)
# np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_ae_history_stage1.npy", train_loss_ae_history)
# np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)


cgkn = torch.load(r"../model/QG(Noisy)_CGKN__dimz256_stage1.pt").to(device)


# # Model Diagnosis in Physical Space
# train_u1 = train_u1.to(device)
# train_u2 = train_u2.to(device)
# 
# with torch.no_grad():
#     train_z_concat = cgkn.autoencoder.encoder(train_u2).view(Ntrain, dim_z)
#     train_u_extended = torch.cat([train_u1, train_z_concat], dim=-1)
#     train_u_extended_pred = cgkn(train_u_extended)
#     train_u1_pred = train_u_extended_pred[:, :dim_u1]
#     train_z_pred = train_u_extended_pred[:, dim_u1:].view(Ntrain, int(dim_z**0.5), int(dim_z**0.5))
#     train_u2_pred = cgkn.autoencoder.decoder(train_z_pred)
# train_u1_err = train_u1 - train_u1_pred
# train_u1_err = (train_u1_err+torch.pi) % (2*torch.pi) - torch.pi
# MSE1 = torch.mean(train_u1_err**2)
# print("MSE1:", MSE1.item())
# MSE2 = nnF.mse_loss(train_u2[1:], train_u2_pred[:-1])
# print("MSE2:", MSE2.item())
# 
# train_u1 = train_u1.to("cpu")
# train_u2 = train_u2.to("cpu")


#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################

train_u1 = train_u1.to(device)
train_u2 = train_u2.to(device)
with torch.no_grad():
    train_z_concat = cgkn.autoencoder.encoder(train_u2).view(Ntrain, dim_z)
    train_u_extended = torch.cat([train_u1, train_z_concat], dim=-1)
    train_u_extended_pred = cgkn(train_u_extended)
    train_u1_pred = train_u_extended_pred[:, :dim_u1]
train_u1_err = train_u1 - train_u1_pred
train_u1_err = (train_u1_err+torch.pi) % (2*torch.pi) - torch.pi
sigma_hat = torch.zeros(dim_u1 + dim_z)
sigma_hat[:dim_u1] = torch.sqrt(torch.mean(train_u1_err**2, dim=0))
sigma_hat[dim_u1:] = 0.1 # sigma2 is set manually
train_u1 = train_u1.to("cpu")
train_u2 = train_u2.to("cpu")
# sigma_hat = torch.sqrt( torch.mean( (train_u_extended[1:] - train_u_extended_pred[:-1])**2, dim=0 ) )


def CGFilter(cgkn, sigma, u1, mu0, R0):
    # u1: (t, x, 1)
    # mu0: (x, 1)
    # R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_u1 = u1.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    mu_pred = torch.zeros((Nt, dim_z, 1)).to(device)
    R_pred = torch.zeros((Nt, dim_z, dim_z)).to(device)
    for n in range(Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n].T)]
        mu1 = f2 + g2@mu0 + g2@R0@g1.T@torch.inverse(s1@s1.T+g1@R0@g1.T)@(u1[n]-f1-g1@mu0)
        R1 = g2@R0@g2.T + s2@s2.T - g2@R0@g1.T@torch.inverse(s1@s1.T+g1@R0@g1.T)@g1@R0@g2.T
        mu_pred[n] = mu1
        R_pred[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_pred, R_pred)


########################################################
################# Train cgkn (Stage2)  #################
########################################################

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = 2
long_steps = 800
cut_point = 50

Niters = 10000
train_batch_size = 100
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
# Re-initialize Model
autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for itr in range(1, Niters+1):
    start_time = time.time()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u1_short = torch.stack([train_u1[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (2, 100, 256)
    u2_short = torch.stack([train_u2[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (2, 100, 128, 128, 2)

    # AutoEncoder
    z_short = cgkn.autoencoder.encoder(u2_short.view(-1, *u2_short.shape[2:]))
    u2_ae_short = cgkn.autoencoder.decoder(z_short).view(*u2_short.shape)
    loss_ae = nnF.mse_loss(u2_short, u2_ae_short)

    # State Prediction
    z_short = z_short.view(short_steps, train_batch_size, *z_short.shape[1:]) # (2, 100, 16, 16)
    z_concat_short = z_short.reshape(short_steps, train_batch_size, dim_z) # (2, 100, 256)
    u_extended0_short = torch.cat([u1_short[0], z_concat_short[0]], dim=-1)
    u_extended_pred_short = torch.zeros(short_steps, train_batch_size, dim_u1+dim_z).to(device)
    u_extended_pred_short[0] = u_extended0_short
    for n in range(1, short_steps):
        u_extended1_short = cgkn(u_extended0_short)
        u_extended_pred_short[n] = u_extended1_short
        u_extended0_short = u_extended1_short
    z_concat_pred_short = u_extended_pred_short[:, :, dim_u1:]
    loss_forecast_z = nnF.mse_loss(z_concat_short[1:], z_concat_pred_short[1:])

    u1_pred_short = u_extended_pred_short[:, :, :dim_u1]
    z_pred_short = z_concat_pred_short.reshape(short_steps, train_batch_size, int(dim_z ** 0.5), int(dim_z ** 0.5)) 
    u2_pred_short = cgkn.autoencoder.decoder(z_pred_short.view(-1, int(dim_z**0.5), int(dim_z**0.5))).view(*u2_short.shape)
    u1_err = u1_short[1:] - u1_pred_short[1:]
    u1_err = (u1_err+torch.pi) % (2*torch.pi) - torch.pi
    mse1_short = torch.mean(u1_err**2)
    mse2_short = nnF.mse_loss(u2_short[1:], u2_pred_short[1:])
    loss_forecast = mse1_short + mse2_short

    # DA
    head_idx_long = torch.from_numpy(np.random.choice(Ntrain-long_steps+1, size=1, replace=False))
    u1_long = train_u1[head_idx_long:head_idx_long + long_steps].to(device)
    u2_long = train_u2[head_idx_long:head_idx_long + long_steps].to(device)
    mu_z_concat_pred_long = CGFilter(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device))[0].squeeze(-1)
    mu_z_pred_long = mu_z_concat_pred_long.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
    loss_da = nnF.mse_loss(u2_long[cut_point:], mu_pred_long)
    loss_total = loss_forecast + loss_da + loss_ae + loss_forecast_z
    if torch.isnan(loss_total):
        print(itr, "nan")
        continue

    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae_history.append(loss_ae.item())
    train_loss_forecast_z_history.append(loss_forecast_z.item())

    end_time = time.time()
    print("itr", itr,
          " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae:", round(loss_ae.item(),4),
          " loss fore z:", round(loss_forecast_z.item(), 4))

torch.save(cgkn, r"../model/QG(Noisy)_CGKN__dimz256_stage2.pt")
np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_forecast_history_stage2.npy", train_loss_forecast_history)
np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_ae_history_stage2.npy", train_loss_ae_history)
np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)
np.save(r"../model/QG(Noisy)_CGKN__dimz256_train_loss_da_history_stage2.npy", train_loss_da_history)


exit()


#############################################
################# Test cgkn #################
#############################################

# CGKN for One-Step Prediction
test_batch_size = 1000
test_tensor = torch.utils.data.TensorDataset(test_u)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u1_pred = torch.zeros_like(test_u1)
test_u2_pred = torch.zeros_like(test_u2)
si = 0
for u in test_loader:
    test_u_batch = u[0].to(device)
    test_u1_batch = test_u_batch[:, indices_gridx_u1, indices_gridy_u1]
    test_u2_batch = test_u_batch[:, indices_gridx_u2, indices_gridy_u2]
    test_u1_concat_batch = test_u1_batch.reshape(-1, dim_u1)
    with torch.no_grad():
        test_z_batch = cgkn.autoencoder.encoder(test_u2_batch)
        test_z_concat_batch = test_z_batch.reshape(-1, dim_z)
        test_u_extended_batch = torch.cat([test_u1_concat_batch, test_z_concat_batch], dim=-1)
        test_u_extended_pred_batch = cgkn(test_u_extended_batch )
        test_u1_concat_pred_batch = test_u_extended_pred_batch[:, :dim_u1]
        test_z_concat_pred_batch = test_u_extended_pred_batch[:, dim_u1:]
        test_u1_pred_batch = test_u1_concat_pred_batch.reshape(-1, len(indices_x_u1), len(indices_y_u1))
        test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5) ))
    ei = si + test_batch_size
    test_u1_pred[si:ei] = test_u1_pred_batch
    test_u2_pred[si:ei] = test_u2_pred_batch
    si = ei
print(nnF.mse_loss(test_u2[1:], test_u2_pred[:-1]).item())
test_u2_original_pred = normalizer.decode(test_u2_pred)
print(nnF.mse_loss(test_u2_original[1:], test_u2_original_pred[:-1]).item())


# CGKN for Multi-Step State Prediction
"""
test_short_steps = 2
mask = torch.ones(Ntest, dtype=torch.bool)
mask[::test_short_steps] = False
test_u0 = test_u[::test_short_steps]
test_u1_0 = test_u0[:, indices_gridx_u1, indices_gridy_u1]
test_u2_0 = test_u0[:, indices_gridx_u2, indices_gridy_u2]
test_u1_0_concat = test_u1_0.reshape(test_u1_0.shape[0], dim_u1)
with torch.no_grad():
    test_z0 = cgkn.autoencoder.encoder(test_u2_0)
test_z0_concat = test_z0.reshape(test_z0.shape[0], dim_z)
test_u_extended0 = torch.cat([test_u1_0_concat, test_z0_concat], dim=-1)
test_u_extended_shortPred = torch.zeros(test_short_steps, test_u0.shape[0], dim_u1+dim_z) # (t, N, x)
test_u_extended_shortPred[0] = test_u_extended0
with torch.no_grad():
    for n in range(test_short_steps-1):
        test_u_extended_shortPred[n+1] = cgkn(test_u_extended_shortPred[n])
test_u1_concat_shortPred = test_u_extended_shortPred[:, :, :dim_u1]
test_z_concat_shortPred = test_u_extended_shortPred[:, :, dim_u1:]
test_u1_shortPred = test_u1_concat_shortPred.reshape(test_short_steps, test_u1_concat_shortPred.shape[1], int(dim_u1**0.5), int(dim_u1**0.5))
test_z_shortPred = test_z_concat_shortPred.reshape(test_short_steps, test_u1_concat_shortPred.shape[1], int(dim_z**0.5), int(dim_z**0.5))
with torch.no_grad():
    test_u2_shortPred = cgkn.autoencoder.decoder( test_z_shortPred.reshape(-1, int(dim_z**0.5), int(dim_z**0.5)) ).reshape(test_short_steps, test_z_shortPred.shape[1], int(dim_u2**0.5), int(dim_u2**0.5))
test_u2_shortPred = test_u2_shortPred.permute(1, 0, 2, 3).reshape(-1, int(dim_u2**0.5), int(dim_u2**0.5))[:Ntest]
nnF.mse_loss(test_u2[mask], test_u2_shortPred[mask])
test_u2_original_shortPred = normalizer.decode(test_u2_shortPred)
nnF.mse_loss(test_u2_original[mask], test_u2_original_shortPred[mask])
"""


# CGKN for Data Assimilation
batch_steps = 1000
test_mu_pred = torch.zeros(Ntest, int(dim_u2**0.5), int(dim_u2**0.5)).to(device)
test_mu_z0 = torch.zeros(dim_z, 1).to(device)
test_R_z0 = 0.01 * torch.eye(dim_z).to(device)
for si in np.arange(0, Ntest, batch_steps):
    with torch.no_grad():
        test_mu_z_concat_pred_batch, test_mu_R_pred_batch = CGFilter(cgkn,
                                                                       sigma_hat.to(device),
                                                                       test_u1.reshape(-1, dim_u1).unsqueeze(-1)[si:si+batch_steps].to(device),
                                                                       test_mu_z0,
                                                                       test_R_z0)
        test_mu_z_pred_batch = test_mu_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        test_mu_pred_batch = cgkn.autoencoder.decoder(test_mu_z_pred_batch)
    test_mu_pred[si:si+batch_steps] = test_mu_pred_batch
    test_mu_z0 = test_mu_z_concat_pred_batch[-1]
    test_R_z0 = test_mu_R_pred_batch[-1]
test_mu_pred = test_mu_pred.to("cpu")
print(nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:]).item())
test_mu_original_pred = normalizer.decode(test_mu_pred)
print(nnF.mse_loss(test_u2_original[cut_point:], test_mu_original_pred[cut_point:]).item())


# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )
