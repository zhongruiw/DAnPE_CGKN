import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)


###############################################
################# Data Import #################
###############################################

# # Simulation Settings: Lt=1000, dt=0.0001
sigma = 10
rho = 28
beta = 8/3
sig1, sig2, sig3 = [0.1, 0.1, 0.1]
Lt = 1000
dt = 0.0001
Nt = int(Lt/dt)
# u = np.zeros((Nt, 3))
# for n in range(Nt-1):
#     u[n+1, 0] = u[n, 0] + (sigma*(u[n, 1] - u[n, 0]))*dt + sig1*np.sqrt(dt)*np.random.randn()
#     u[n+1, 1] = u[n, 1] + (u[n, 0]*(rho-u[n, 2])-u[n, 1])*dt + sig2*np.sqrt(dt)*np.random.randn()
#     u[n+1, 2] = u[n, 2] + (u[n, 0]*u[n, 1] - beta*u[n, 2])*dt + sig3*np.sqrt(dt)*np.random.randn()

# # Sub-Sampling: dt=0.01
# u = u[::100]
u = np.load("./data/Lorenz63.npy")
u = u[:, :2]
u = u + 0.1*np.random.randn(*u.shape)
dt = 0.01
t = np.arange(0, Lt, dt)

# Train / Test
Ntrain = 80000
Ntest = 20000
train_u = torch.from_numpy(u[:Ntrain]).to(torch.float32)
train_t = torch.from_numpy(t[:Ntrain]).to(torch.float32)
test_u = torch.from_numpy(u[-Ntest:]).to(torch.float32)
test_t = torch.from_numpy(t[-Ntest:]).to(torch.float32)

# Observed / Unobserved
indices_u1 = [0]
indices_u2 = [1]

############################################################
################# MemCGKN: MemAE + MemCGN  #################
############################################################

class MemAE(nn.Module):
    def __init__(self, dim_u2, dim_z, len_mem):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_u2*len_mem, 8), nn.SiLU(),
                                     nn.Linear(8, 16), nn.SiLU(),
                                     nn.Linear(16, 32), nn.SiLU(),
                                     nn.Linear(32, 16), nn.SiLU(),
                                     nn.Linear(16, 8), nn.SiLU(),
                                     nn.Linear(8, dim_z))

        self.decoder = nn.Sequential(nn.Linear(dim_z, 8), nn.SiLU(),
                                     nn.Linear(8, 16), nn.SiLU(),
                                     nn.Linear(16, 32), nn.SiLU(),
                                     nn.Linear(32, 16), nn.SiLU(),
                                     nn.Linear(16, 8), nn.SiLU(),
                                     nn.Linear(8, dim_u2*len_mem))

class MemCGN(nn.Module):
    def __init__(self, dim_u1, dim_z, len_mem):
        super().__init__()
        self.input_size = dim_u1*len_mem
        self.f1_size = (dim_u1, 1)
        self.g1_size = (dim_u1, dim_z)
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.output_size = np.prod(self.f1_size) + np.prod(self.g1_size) + np.prod(self.f2_size) + np.prod(self.g2_size)
        self.net = nn.Sequential(nn.Linear(self.input_size, 8), nn.SiLU(),
                                 nn.Linear(8, 16), nn.SiLU(),
                                 nn.Linear(16, 32), nn.SiLU(),
                                 nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(16, 16), nn.SiLU(),
                                 nn.Linear(16, self.output_size))
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        f1 = out[:, :np.prod(self.f1_size)].reshape(batch_size, *self.f1_size)
        g1 = out[:, np.prod(self.f1_size):np.prod(self.f1_size)+np.prod(self.g1_size)].reshape(batch_size, *self.g1_size)
        f2 = out[:, np.prod(self.f1_size)+np.prod(self.g1_size):np.prod(self.f1_size)+np.prod(self.g1_size)+np.prod(self.f2_size)].reshape(batch_size, *self.f2_size)
        g2 = out[:, np.prod(self.f1_size)+np.prod(self.g1_size)+np.prod(self.f2_size):].reshape(batch_size, *self.g2_size)
        return [f1, g1, f2, g2]

class MemCGKN(nn.Module):
    def __init__(self, mem_ae, mem_cgn):
        super().__init__()
        self.mem_ae = mem_ae
        self.mem_cgn = mem_cgn

    def forward(self, u1_mem, z):
        # Matrix Form Computation
        f1, g1, f2, g2 = self.mem_cgn(u1_mem)
        z = z.unsqueeze(-1)
        u1_pred = f1 + g1@z
        z_pred = f2 + g2@z
        return [u1_pred.squeeze(-1), z_pred.squeeze(-1)]


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
dim_z = 15
len_mem = 5

# Stage1: Train MemCGKN with loss_forecast + loss_ae + loss_forecast_z
Niters = 160000
train_batch_size = 1000
train_loss_forecast_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

mem_ae = MemAE(dim_u2, dim_z, len_mem).to(device)
mem_cgn = MemCGN(dim_u1, dim_z, len_mem).to(device)
mem_cgkn = MemCGKN(mem_ae, mem_cgn).to(device)
optimizer = torch.optim.Adam(mem_cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
for itr in range(1, Niters+1):
    start_time = time.time()

    indices_initial = torch.randint(0, Ntrain-len_mem, (train_batch_size,))
    u_mem = torch.stack([train_u[indices_initial+n] for n in range(len_mem)], dim=1)
    u1_mem = u_mem[:, :, indices_u1]
    u2_mem = u_mem[:, :, indices_u2]
    u_mem_next = torch.stack([train_u[indices_initial+n+1] for n in range(len_mem)], dim=1)
    u2_mem_next = u_mem_next[:, :, indices_u2]
    u_next = u_mem_next[:, -1]

    # MemAE
    z = mem_cgkn.mem_ae.encoder(u2_mem.view(u2_mem.shape[0], len_mem*dim_u2))
    u2_mem_ae = mem_cgkn.mem_ae.decoder(z).view(z.shape[0], len_mem, dim_u2)
    loss_ae = nnF.mse_loss(u2_mem, u2_mem_ae)

    #  State Forecast
    u1_pred, z_pred = mem_cgkn(u1_mem.view(u1_mem.shape[0], len_mem*dim_u1), z)
    u2_mem_pred = mem_cgkn.mem_ae.decoder(z_pred).view(z_pred.shape[0], len_mem, dim_u2)
    u2_pred = u2_mem_pred[:, -1]

    u_pred = torch.zeros_like(u_next)
    u_pred[:, indices_u1] = u1_pred
    u_pred[:, indices_u2] = u2_pred
    loss_forecast = nnF.mse_loss(u_next, u_pred)

    #  State Forecast z
    z_next = mem_cgkn.mem_ae.encoder(u2_mem_next.view(u2_mem_next.shape[0], len_mem*dim_u2))
    loss_forecast_z = nnF.mse_loss(z_next, z_pred)

    loss_total = loss_forecast + loss_ae + loss_forecast_z
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_ae_history.append(loss_ae.item())
    train_loss_forecast_z_history.append(loss_forecast_z.item())

    end_time = time.time()
    print("itr:", itr,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss ae:", round(loss_ae.item(), 4),
          " loss fore z:", round(loss_forecast_z.item(), 4))
# """

# torch.save(mem_cgkn, r"./model/Model_MemCGKN/L63_MemCGKN_dimz15_stage1.pt")
# np.save(r"./model/Model_MemCGKN/L63_MemCGKN_dimz15_train_loss_forecast_history_stage1.npy", train_loss_forecast_history)
# np.save(r"./model/Model_MemCGKN/L63_MemCGKN_dimz15_train_loss_ae_history_stage1.npy", train_loss_ae_history)
# np.save(r"./model/Model_MemCGKN/L63_MemCGKN_dimz15_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

mem_cgkn = torch.load(r"./model/Model_MemCGKN/L63_MemCGKN_dimz15_stage1.pt")

# # Model Diagnosis in Physical Space
# mem_cgkn.to("cpu")
# train_u_mem = train_u.unfold(dimension=0, size=len_mem, step=1).permute(0, 2, 1)
# train_u1_mem = train_u_mem[:, :, indices_u1]
# train_u2_mem = train_u_mem[:, :, indices_u2]
# with torch.no_grad():
#     train_z = mem_cgkn.mem_ae.encoder(train_u2_mem.view(train_u2_mem.shape[0], len_mem*dim_u2))
#     train_u1_pred, train_z_pred = mem_cgkn(train_u1_mem.view(train_u1_mem.shape[0], len_mem*dim_u1), train_z)
#     train_u2_mem_pred = mem_cgkn.mem_ae.decoder(train_z_pred).view(train_z_pred.shape[0], len_mem, dim_u2)
#     train_u2_pred = train_u2_mem_pred[:, -1]
# train_u_pred = torch.zeros_like(train_u[len_mem:])
# train_u_pred[:, indices_u1] = train_u1_pred[:-1]
# train_u_pred[:, indices_u2] = train_u2_pred[:-1]
# nnF.mse_loss(train_u[len_mem:], train_u_pred)
# mem_cgkn.to(device)



#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################
mem_cgkn.to("cpu")
train_u_mem = train_u.unfold(dimension=0, size=len_mem, step=1).permute(0, 2, 1)
train_u1_mem = train_u_mem[:, :, indices_u1]
train_u2_mem = train_u_mem[:, :, indices_u2]
with torch.no_grad():
    train_z = mem_cgkn.mem_ae.encoder(train_u2_mem.view(train_u2_mem.shape[0], len_mem*dim_u2))
    train_u1_pred, train_z_pred = mem_cgkn(train_u1_mem.view(train_u1_mem.shape[0], len_mem*dim_u1), train_z)
sigma_hat = torch.zeros(dim_u1 + dim_z)
sigma_hat[:dim_u1] = torch.mean((train_u[len_mem:, indices_u1] - train_u1_pred[:-1])**2,dim=0)**0.5
sigma_hat[dim_u1:] = 0.1 # sigma2 is set manually
mem_cgkn.to(device)


def CGFilter(mem_cgkn, len_mem, sigma, u1, mu0, R0):
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
    mu_pred[0] = mu0
    R_pred[0] = R0
    for n in range(len_mem, Nt):
        u1_mem = u1[n-len_mem:n].squeeze(-1)
        f1, g1, f2, g2 = [e.squeeze(0) for e in mem_cgkn.mem_cgn(u1_mem.view(1, len_mem*dim_u1))]
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

# short_steps = 2
long_steps = 2000
cut_point = 20

Niters = 10000
train_batch_size = 500
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
# Re-initialize Model
mem_ae = MemAE(dim_u2, dim_z, len_mem).to(device)
mem_cgn = MemCGN(dim_u1, dim_z, len_mem).to(device)
mem_cgkn = MemCGKN(mem_ae, mem_cgn).to(device)
optimizer = torch.optim.Adam(mem_cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for itr in range(1, Niters+1):
    start_time = time.time()

    indices_initial = torch.randint(0, Ntrain-len_mem, (train_batch_size,))
    u_mem = torch.stack([train_u[indices_initial+n] for n in range(len_mem)], dim=1)
    u1_mem = u_mem[:, :, indices_u1]
    u2_mem = u_mem[:, :, indices_u2]
    u_mem_next = torch.stack([train_u[indices_initial+n+1] for n in range(len_mem)], dim=1)
    u2_mem_next = u_mem_next[:, :, indices_u2]
    u_next = u_mem_next[:, -1]

    # MemAE
    z = mem_cgkn.mem_ae.encoder(u2_mem.view(u2_mem.shape[0], len_mem*dim_u2))
    u2_mem_ae = mem_cgkn.mem_ae.decoder(z).view(z.shape[0], len_mem, dim_u2)
    loss_ae = nnF.mse_loss(u2_mem, u2_mem_ae)

    #  State Forecast
    u1_pred, z_pred = mem_cgkn(u1_mem.view(u1_mem.shape[0], len_mem*dim_u1), z)
    u2_mem_pred = mem_cgkn.mem_ae.decoder(z_pred).view(z_pred.shape[0], len_mem, dim_u2)
    u2_pred = u2_mem_pred[:, -1]

    u_pred = torch.zeros_like(u_next)
    u_pred[:, indices_u1] = u1_pred
    u_pred[:, indices_u2] = u2_pred
    loss_forecast = nnF.mse_loss(u_next, u_pred)

    #  State Forecast z
    z_next = mem_cgkn.mem_ae.encoder(u2_mem_next.view(u2_mem_next.shape[0], len_mem*dim_u2))
    loss_forecast_z = nnF.mse_loss(z_next, z_pred)

    # DA
    head_idx_long = torch.from_numpy(np.random.choice(Ntrain-long_steps+1, size=1, replace=False))
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    mu_z_pred_long = CGFilter(mem_cgkn, len_mem, sigma_hat.to(device), u_long[:, indices_u1].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device))[0].squeeze(-1)
    mu_mem_pred_long = mem_cgkn.mem_ae.decoder(mu_z_pred_long[cut_point:]).view(mu_z_pred_long[cut_point:].shape[0], len_mem, dim_u2)
    mu_pred_long = mu_mem_pred_long[:, -1]
    loss_da = nnF.mse_loss(u_long[cut_point:, indices_u2], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae + loss_forecast_z
    if torch.isnan(loss_total):
        print(itr, "nan")
        continue
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

torch.save(mem_cgkn, r"./model/Model_CGKN/L63_MemCGKN_dimz15_stage2.pt")
np.save(r"./model/Model_CGKN/L63_MemCGKN_dimz15_train_loss_forecast_history_stage2.npy", train_loss_forecast_history)
np.save(r"./model/Model_CGKN/L63_MemCGKN_dimz15_train_loss_da_history_stage2.npy", train_loss_da_history)
np.save(r"./model/Model_CGKN/L63_MemCGKN_dimz15_train_loss_ae_history_stage2.npy", train_loss_ae_history)
np.save(r"./model/Model_CGKN/L63_MemCGKN_dimz15_train_loss_forecast_z_history_stage2.npy", train_loss_forecast_z_history)


#####################################################################################
################# DA Uncertainty Quantification via Residual Analysis ###############
#####################################################################################

# Data Assimilation of Train Data
with torch.no_grad():
    train_mu_z_pred, train_R_z_pred = CGFilter(cgkn,
                                               sigma_hat.to(device),
                                               train_u[:, indices_u1].unsqueeze(-1).to(device),
                                               torch.zeros(dim_z, 1).to(device),
                                               0.01 * torch.eye(dim_z).to(device))
    train_mu_pred = cgkn.autoencoder.decoder(train_mu_z_pred.squeeze(-1)).cpu()

del train_mu_z_pred, train_R_z_pred
gc.collect()
torch.cuda.empty_cache()

# Target Variable: Residual (std of posterior mean)
train_mu_std = torch.abs(train_u[cut_point:, indices_u2] - train_mu_pred[cut_point:])

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_u1, 16), nn.SiLU(),
                                 nn.Linear(16, 32), nn.SiLU(),
                                 nn.Linear(32, 32), nn.SiLU(),
                                 nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(16, dim_u2))

    def forward(self, x):
        out = self.net(x)
        return out

epochs = 1000
train_batch_size = 500
train_tensor = torch.utils.data.TensorDataset(train_u[cut_point:, indices_u1], train_mu_std)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_uncertainty_history = []

uncertainty_net = UncertaintyNet(dim_u1, dim_u2).to(device)
optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_uncertainty = 0.
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = uncertainty_net(x)
        loss = nnF.mse_loss(y, out)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss_uncertainty += loss.item()
    train_loss_uncertainty /= train_num_batches
    train_loss_uncertainty_history.append(train_loss_uncertainty)
    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss uncertainty:", round(train_loss_uncertainty, 4))

# torch.save(uncertainty_net, path_abs + r"/Models/Model_CGKN/KSE_UQNet.pt")
# np.save(path_abs + r"/Models/Model_CGKN/KSE_UQNet_train_loss_uncertainty_history.npy", train_loss_uncertainty_history)

uncertainty_net = torch.load(path_abs + r"/Models/Model_CGKN/KSE_UQNet.pt").to(device)


#############################################
################# Test cgkn #################
#############################################
cgkn.cpu()
uncertainty_net.cpu()
device = next(cgkn.parameters()).device

# CGKN for One-Step Prediction
with torch.no_grad():
    test_z = cgkn.autoencoder.encoder(test_u[:, indices_u2].to(device))
test_u_extended = torch.cat([test_u[:, indices_u1].to(device), test_z], dim=-1)
with torch.no_grad():
    test_u_extended_pred = cgkn(test_u_extended)
    test_u2_pred = cgkn.autoencoder.decoder(test_u_extended_pred[:, dim_u1:])
test_u_pred = torch.zeros_like(test_u)
test_u_pred[:, indices_u1] = test_u_extended_pred[:, :dim_u1].cpu()
test_u_pred[:, indices_u2] = test_u2_pred.cpu()
nnF.mse_loss(test_u[1:], test_u_pred[:-1]).item()


# CGKN for Data Assimilation
st = time.time()
test_mu_pred = torch.zeros(1000, 128)
with torch.no_grad():
    test_mu_z_pred, test_R_z_pred = CGFilter(cgkn,
                                             sigma_hat.to(device),
                                             test_u[:, indices_u1].unsqueeze(-1).to(device),
                                             torch.zeros(dim_z, 1).to(device),
                                             0.01 * torch.eye(dim_z).to(device))
    test_mu_pred[:, indices_u2] = cgkn.autoencoder.decoder(test_mu_z_pred.squeeze(-1)).cpu()
test_mu_pred[:, indices_u1] = test_u[:, indices_u1]
nnF.mse_loss(test_u[cut_point:], test_mu_pred[cut_point:]).item()

# uncertainty_net for Uncertainty Quantification
test_mu_std_pred = torch.zeros(1000, 128)
with torch.no_grad():
    test_mu_std_pred[:, indices_u2] = uncertainty_net(test_u[:, indices_u1].to(device)).cpu()
test_mu_std_pred[:, indices_u1] = 0
et = time.time()
time_DA = et - st
print("DA time:", time_DA)
# np.savez(path_abs + "/Data/KSE(Noisy)_CGKN_DA.npz", mean=test_mu_pred, std=test_mu_std_pred)



# CGKN for State Prediction (Advanced Version)
test_short_steps = 20
test_u_initial = test_u[::test_short_steps]
with torch.no_grad():
    test_z_initial = cgkn.autoencoder.encoder(test_u_initial[:, indices_u2])
test_u_extended_initial = torch.cat([test_u_initial[:, indices_u1], test_z_initial], dim=-1)
test_u_extended_shortPred = torch.zeros(test_short_steps, test_u_initial.shape[0], dim_u1+dim_z) # (t, N, x)
test_u_extended_shortPred[0] = test_u_extended_initial
with torch.no_grad():
    for n in range(test_short_steps-1):
        test_u_extended_shortPred[n+1] = cgkn(test_u_extended_initial)
        test_z_next = test_u_extended_shortPred[n+1, :, dim_u1:]
        test_u2_next = cgkn.autoencoder.decoder(test_z_next)
        test_z_next_ae = cgkn.autoencoder.encoder(test_u2_next)
        test_u_extended_initial = torch.cat([test_u_extended_shortPred[n+1, :, :dim_u1], test_z_next_ae], dim=-1)
    test_u2_shortPred = cgkn.autoencoder.decoder(test_u_extended_shortPred[:, :, dim_u1:])
test_u_shortPred = torch.zeros(test_short_steps, *test_u_initial.shape)
test_u_shortPred[:, :, indices_u1] = test_u_extended_shortPred[:, :, :dim_u1]
test_u_shortPred[:, :, indices_u2] = test_u2_shortPred
test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(-1, 128)[:Ntest]
mask = torch.ones(Ntest, dtype=torch.bool)
mask[::test_short_steps] = False
nnF.mse_loss(test_u[mask], test_u_shortPred[mask])
# np.save(path_abs + r"/Data/KSE(Noisy)_CGKN_SF_4000initial_20steps", test_u_shortPred[:20])



# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )


# # CGKN for Multi-Step Prediction (Naive)
# test_short_steps = 10
# test_u0 = test_u[::test_short_steps]
# mask = torch.ones(Ntest, dtype=torch.bool)
# mask[::test_short_steps] = False
# with torch.no_grad():
#     test_z0 = cgkn.autoencoder.encoder(test_u0[:, indices_u2])
# test_u_extended0 = torch.cat([test_u0[:, indices_u1], test_z0], dim=-1)
# test_u_extended_shortPred = torch.zeros(test_short_steps, test_u0.shape[0], dim_u1+dim_z) # (t, N, x)
# test_u_extended_shortPred[0] = test_u_extended0
# with torch.no_grad():
#     for n in range(test_short_steps-1):
#         test_u_extended_shortPred[n+1] = cgkn(test_u_extended_shortPred[n])
#     test_u2_shortPred = cgkn.autoencoder.decoder(test_u_extended_shortPred[:, :, dim_u1:])
# test_u_shortPred = torch.zeros(test_short_steps, *test_u0.shape)
# test_u_shortPred[:, :, indices_u1] = test_u_extended_shortPred[:, :, :dim_u1]
# test_u_shortPred[:, :, indices_u2] = test_u2_shortPred
# test_u_shortPred = test_u_shortPred.permute(1, 0, 2).reshape(-1, test_u_shortPred.shape[-1])[:Ntest]
# nnF.mse_loss(test_u[mask], test_u_shortPred[mask]).item()

