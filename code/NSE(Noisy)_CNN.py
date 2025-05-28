import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

path_abs = r"C:\Users\chenc\CodeProject\Discrete CGKN\NSE"
device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

###############################################
################# Data Import #################
###############################################

# Simulation Settings: Lt=1000, Lx=1, Ly=1, dt=0.001, dx=1/256, dy=1/256
Lt = 1000.
Lx = 1.
Ly = 1.

# Data Resolution: dt=0.01, dx=1/64, dy=1/64
u = np.load(path_abs + "/Data/NSE_Data.npy")
t = np.arange(0, Lt, 0.01)
x = np.arange(0, Lx, 1/64)
y = np.arange(0, Ly, 1/64)
gridx, gridy = np.meshgrid(x, y, indexing="ij")


# Train / Test
Ntrain = 80000
Ntest = 20000
train_u_original = torch.from_numpy(u[:Ntrain]).to(torch.float32)
train_t = torch.from_numpy(t[:Ntrain]).to(torch.float32)
test_u_original = torch.from_numpy(u[-Ntest:]).to(torch.float32)
test_t = torch.from_numpy(t[-Ntest:]).to(torch.float32)


# Normalization
class Normalizer:
    def __init__(self, x, eps=1e-9):
        # x is in the shape tensor (N, x, y)
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x*(self.std.to(x.device) + self.eps) + self.mean.to(x.device)
normalizer = Normalizer(train_u_original)
train_u = normalizer.encode(train_u_original)
test_u = normalizer.encode(test_u_original)


################################################################
################# CNN: Unit Solution Operator  #################
################################################################

# class SolNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.seq = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
#                                  nn.Conv2d(64, 1, 3, 1, 1))
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         out = self.seq(x)
#         out = out.squeeze(1)
#         return out

class SolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 1, 3, 1, 1))
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.seq(x)
        out = out.squeeze(1)
        return out

##############################################
################# Train CNN  #################
##############################################

epochs = 1000
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
    start_time = time.time()

    train_loss_forecast = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)
        u_pred = solnet(u)
        loss_forecast = nnF.mse_loss(u_next, u_pred)
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss_forecast.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_forecast += loss_forecast.item()
    train_loss_forecast /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4))

torch.save(solnet, path_abs + r"/Models/Model_CNN/NSE(Noisy)_CNN(411073).pt")
np.save(path_abs + r"/Models/Model_CNN/NSE(Noisy)_CNN(411073)_train_loss_forecast_history.npy", train_loss_forecast_history)

# solnet = torch.load(path_abs + r"/Models/Model_CNN/NSE(Noisy)_CNN.pt").to(device)


# One-step Prediction
test_batch_size = 1000
test_tensor = torch.utils.data.TensorDataset(test_u)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u_pred = torch.zeros_like(test_u)
si = 0
for u in test_loader:
    test_u_batch = u[0].to(device)
    with torch.no_grad():
        test_u_pred[si:si+test_batch_size] = solnet(test_u_batch)
    si += test_batch_size
print(nnF.mse_loss(test_u[1:], test_u_pred[:-1]).item())
test_u_original_pred = normalizer.decode(test_u_pred)
print(nnF.mse_loss(test_u_original[1:], test_u_original_pred[:-1]).item())


# Multi-step Prediction
test_short_steps = 10
mask = torch.ones(Ntest, dtype=torch.bool)
mask[::test_short_steps] = False
test_u_initial = test_u[::test_short_steps]
test_batch_size = 100
test_tensor = torch.utils.data.TensorDataset(test_u_initial)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u_shortPred = torch.zeros(test_short_steps*test_u_initial.shape[0], 64, 64)
si = 0
for u in test_loader:
    test_u_initial_batch = u[0].to(device)
    test_u_shortPred_batch = torch.zeros(test_short_steps, *test_u_initial_batch.shape).to(device)
    test_u_shortPred_batch[0] = test_u_initial_batch
    with torch.no_grad():
        for n in range(test_short_steps-1):
            test_u_shortPred_batch[n+1] = solnet(test_u_shortPred_batch[n])
    test_u_shortPred_batch = test_u_shortPred_batch.permute(1, 0, 2, 3).reshape(-1, 64, 64)
    test_u_shortPred[si:si+test_u_shortPred_batch.shape[0]] = test_u_shortPred_batch
    si = si+test_u_shortPred_batch.shape[0]
test_u_shortPred = test_u_shortPred[:Ntest]
print(nnF.mse_loss(test_u[mask], test_u_shortPred[mask]).item())
test_u_original_shortPred = normalizer.decode(test_u_shortPred)
print(nnF.mse_loss(test_u_original[mask], test_u_original_shortPred[mask]).item())





# Multi-step Prediction (Data start from 950)
si = 15000
steps = 11
u_shortPred = torch.zeros(steps, 1, 64, 64).to(device)
u_shortPred[0] = test_u[[si]].to(device)
with torch.no_grad():
    for n in range(steps-1):
        u_shortPred[n+1] = solnet(u_shortPred[n])
u_original_shortPred = normalizer.decode(u_shortPred.squeeze(1).cpu())
# np.save(path_abs + r"/Data/NSE(Noisy)_CNN_950initial_10steps.npy", u_original_shortPred)
