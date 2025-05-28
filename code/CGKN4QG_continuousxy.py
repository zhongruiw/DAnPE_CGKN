import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time
from torchviz import make_dot

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_sigmaxy01_continuousxy.npz")

pos = QG_Data["xy_obs"]
psi = QG_Data["psi_noisy"]

# # exclude jumps in tracer postions
# threshold = 1.5 * np.pi
# jumps = np.abs(np.diff(pos, axis=1))
# jump_flags = np.any(jumps > threshold, axis=(2,3))
# jump_flags[:, -1] = 1
# hitting_time = np.zeros(200)
# for i in range(200):
#     hitting_time[i] = np.where(jump_flags[i,:])[0][0]
# select_ind = np.where(hitting_time==49)[0]
# pos = pos[select_ind] # shape (Nens, Nt, L, 2)
# psi = psi[select_ind] # shape (Nens, Nt, Nx, Nx, 2)

u1 = torch.tensor(pos, dtype=torch.float).view(pos.shape[0], pos.shape[1], -1) # shape (Nens, Nt, 2*L)
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nens, Nt, Nx, Nx, 2)

# Train / Test
Ntrain = 1000 # Nens 0-1000
Ntest = 250 # Nens 1000-1250

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
        # out shape(t, d1, d2)
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
        # out shape(t, y, x, 2)
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
        # Homogeneous Tracers
        self.input_size = 2
        self.output_size = self.input_size + self.input_size*dim_z
        self.net = nn.Sequential(nn.Linear(self.input_size, 8), nn.SiLU(),
                                 nn.Linear(8, 16), nn.SiLU(),
                                 nn.Linear(16, 32), nn.SiLU(),
                                 nn.Linear(32, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(16, self.output_size))
        self.input_size2 = 2*128
        self.output_size2 = dim_z + dim_z*dim_z
        self.net2 = nn.Sequential(nn.Linear(self.input_size2, 512), nn.SiLU(),
                         nn.Linear(512, 1024), nn.SiLU(),
                         nn.Linear(1024, 512), nn.SiLU(),
                         nn.Linear(512, self.output_size2))
        # self.f2_param = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        # self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))

    def forward(self, x):
        # x shape(N, 256)
        batch_size = x.shape[0]

        out2 = self.net2(x) # shape(N, 256+256*256)
        f2 = out2[:, :self.input_size2].reshape(batch_size, *self.f2_size)
        g2 = out2[:, self.input_size2:].reshape(batch_size, *self.g2_size)

        x = x.view(batch_size, -1, self.input_size) # shape(N, 128, 2)
        out = self.net(x) # shape(N, 128, 2+2*256)
        f1 = out[:, :, :self.input_size].reshape(batch_size, *self.f1_size)
        g1 = out[:, :, self.input_size:].reshape(batch_size, *self.g1_size)
        # f2 = self.f2_param.repeat(batch_size, 1, 1)
        # g2 = self.g2_param.repeat(batch_size, 1, 1)
        return [f1, g1, f2, g2]

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u_extended):
        # Matrix-form Computation
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
epochs = 100
# train_batch_size = 400
# train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
# train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
# train_num_batches = len(train_loader)
train_num_batches = Ntrain
Niters = epochs * train_num_batches
train_loss_forecast_u1_history = []
train_loss_forecast_u2_history = []
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

    train_loss_forecast_u1 = 0.
    train_loss_forecast_u2 = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    # for u1_initial, u2_initial, u1_next, u2_next in train_loader:
    for iens in range(Ntrain):
        u1_initial = train_u1[iens, :-1] # shape (50, 256)
        u1_next = train_u1[iens, 1:] # shape (50, 256)
        u2_initial = train_u2[iens, :-1] # shape (50, 128, 128, 2)
        u2_next = train_u2[iens, 1:] # shape (50, 128, 128, 2)
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
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_next_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_next_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_next_pred)

        loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z

        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()

    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_forecast_u1_history.append(train_loss_forecast_u1)
    train_loss_forecast_u2_history.append(train_loss_forecast_u2)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast_u1, 4), round(train_loss_forecast_u2, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))

# torch.save(cgkn, r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_stage1.pt")
# np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_u1_history_stage1.npy", train_loss_forecast_u1_history)
# np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_u2_history_stage1.npy", train_loss_forecast_u2_history)
# np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_ae_history_stage1.npy", train_loss_ae_history)
# np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

"""
cgkn = torch.load(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_stage1.pt").to(device)

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

# Move data to device first if it fits, or load per batch if necessary
train_u1 = train_u1.to(device)
train_u2 = train_u2.to(device)
train_u1_preds = []
with torch.no_grad():
    for iens in range(0, Ntrain):
        batch_u1 = train_u1[iens]
        batch_u2 = train_u2[iens]
        batch_z_concat = cgkn.autoencoder.encoder(batch_u2).view(batch_u2.size(0), dim_z)
        batch_u_extended = torch.cat([batch_u1, batch_z_concat], dim=-1)
        batch_u_extended_pred = cgkn(batch_u_extended)
        batch_u1_pred = batch_u_extended_pred[:, :dim_u1]
        train_u1_preds.append(batch_u1_pred)
    train_u1_pred = torch.stack(train_u1_preds, dim=0)
sigma_hat = torch.zeros(dim_u1 + dim_z)
sigma_hat[:dim_u1] = torch.sqrt(torch.mean((train_u1[:, 1:] - train_u1_pred[:, :-1])**2, dim=(0,1))).to("cpu")
sigma_hat[dim_u1:] = 0.5  # sigma2 manually set
train_u1 = train_u1.to("cpu")
train_u2 = train_u2.to("cpu")
del train_u1_pred, train_u1_preds, batch_u1_pred, batch_u_extended_pred, batch_u_extended, batch_z_concat
torch.cuda.empty_cache()

# np.save('../data/sigma_u1_f2g2.npy', sigma_hat[:dim_u1])


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
long_steps = 50
cut_point = 10

epochs = 100
train_num_batches = Ntrain
Niters = epochs * train_num_batches
train_loss_forecast_u1_history = []
train_loss_forecast_u2_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
# Re-initialize Model
autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
for ep in range(epochs):
    for iens in range(Ntrain):
        start_time = time.time()

        u1_initial = train_u1[iens, :-1] # shape (50, 256)
        u1_next = train_u1[iens, 1:] # shape (50, 256)
        u2_initial = train_u2[iens, :-1] # shape (50, 128, 128, 2)
        u2_next = train_u2[iens, 1:] # shape (50, 128, 128, 2)
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
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_next_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_next_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_next_pred)

        # DA
        u1_long = train_u1[iens].to(device)
        u2_long = train_u2[iens].to(device)
        mu_z_concat_pred_long = CGFilter(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
        mu_z_pred_long = mu_z_concat_pred_long.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
        loss_da = nnF.mse_loss(u2_long[cut_point:], mu_pred_long)

        loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z #+ loss_da
        if torch.isnan(loss_total):
            print(iens, "nan")
            continue

        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_forecast_u1_history.append(loss_forecast_u1.item())
        train_loss_forecast_u2_history.append(loss_forecast_u2.item())
        train_loss_da_history.append(loss_da.item())
        train_loss_ae_history.append(loss_ae.item())
        train_loss_forecast_z_history.append(loss_forecast_z.item())

        end_time = time.time()
        print("itr", int(iens + train_num_batches*ep),
              " time:", round(end_time-start_time, 4),
              " loss fore:", round(loss_forecast_u1.item(), 4), round(loss_forecast_u2.item(), 4),
              " loss da:", round(loss_da.item(),4),
              " loss ae:", round(loss_ae.item(),4),
              " loss fore z:", round(loss_forecast_z.item(), 4))
 
# """

torch.save(cgkn, r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_stage2.pt")
np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_u1_history_stage2.npy", train_loss_forecast_u1_history)
np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_u2_history_stage2.npy", train_loss_forecast_u2_history)
np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_ae_history_stage2.npy", train_loss_ae_history)
np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)
np.save(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_train_loss_da_history_stage2.npy", train_loss_da_history)


# cgkn = torch.load(r"../model/QG(Noisy)_sigmaxy01_continuousxy_CGKN_dimz256_f2g2_stage2.pt").to(device)


############################################
################ Test cgkn #################
############################################

# CGKN for One-Step Prediction
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
test_u1_preds = []
test_u2_preds = []
with torch.no_grad():
    for iens in range(Ntest):
        batch_u1 = test_u1[iens] # shape (50, 256)
        batch_u2 = test_u2[iens] # shape (50, 128, 128, 2)
        batch_z_concat = cgkn.autoencoder.encoder(batch_u2).view(batch_u2.size(0), dim_z)
        batch_u_extended = torch.cat([batch_u1, batch_z_concat], dim=-1)
        batch_u_extended_pred = cgkn(batch_u_extended)
        batch_u1_pred = batch_u_extended_pred[:, :dim_u1]
        batch_z_pred = batch_u_extended_pred[:, dim_u1:].view(batch_u_extended_pred.size(0), int(dim_z**0.5), int(dim_z**0.5))
        batch_u2_pred = cgkn.autoencoder.decoder(batch_z_pred)
        test_u1_preds.append(batch_u1_pred)
        test_u2_preds.append(batch_u2_pred)
    test_u1_pred = torch.stack(test_u1_preds, dim=0)
    test_u2_pred = torch.stack(test_u2_preds, dim=0)
MSE1 = nnF.mse_loss(test_u1[:, 1:], test_u1_pred[:, :-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[:, 1:], test_u2_pred[:, :-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")

np.save(r"../data/CGKN_sigmaxy01_continuousxy_f2g2_xy_unit_OneStepPrediction.npy", test_u1_pred.to("cpu"))
np.save(r"../data/CGKN_sigmaxy01_continuousxy_f2g2_psi_OneStepPrediction.npy", test_u2_pred.to("cpu"))


# CGKN for Data Assimilation
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
test_mu_preds = []
with torch.no_grad():
    for iens in range(Ntest):
        batch_u1 = test_u1[iens] # shape (50, 256)
        batch_u2 = test_u2[iens] # shape (50, 128, 128, 2)
        batch_mu_z_concat_pred = CGFilter(cgkn, sigma_hat.to(device), batch_u1.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
        batch_mu_z_pred = batch_mu_z_concat_pred.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        batch_mu_pred = cgkn.autoencoder.decoder(batch_mu_z_pred)
        test_mu_preds.append(batch_mu_pred)
    test_mu_pred = torch.stack(test_mu_preds, dim=0)
MSE2_DA = nnF.mse_loss(test_u2[:, cut_point:], test_mu_pred[:, cut_point:])
print("MSE2_DA:", MSE2_DA.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/CGKN_sigmaxy01_continuousxy_f2g2_psi_DA.npy", test_mu_pred.to("cpu"))


# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )
