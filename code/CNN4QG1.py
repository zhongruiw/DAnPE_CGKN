import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import time

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data.npz")
pos = QG_Data["xy_obs"]
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1)
psi = QG_Data["psi_noisy"]
u1 = torch.tensor(pos_unit, dtype=torch.float) # shape (Nt, L, 4), keep tracers parallel
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 64, 64, 2)

# Train / Test
Ntrain = 40000
Nval = 5000
Ntest = 5000
L_total = 1024 # total number of tracers in training dataset
L = 128 # number of tracers used in data assimilation

train_u1 = u1[:Ntrain]
train_u2 = u2[:Ntrain]
val_u1 = u1[Ntrain:Ntrain+Nval, :L]
val_u2 = u2[Ntrain:Ntrain+Nval]
test_u1 = u1[Ntrain+Nval:Ntrain+Nval+Ntest, :L]
test_u2 = u2[Ntrain+Nval:Ntrain+Nval+Ntest]
val_u1 = val_u1.to(device)
val_u2 = val_u2.to(device)

################################################################
############################### CNN ############################
################################################################
def unit2xy(xy_unit):
    cos0 = xy_unit[..., 0]
    sin0 = xy_unit[..., 1]
    cos1 = xy_unit[..., 2]
    sin1 = xy_unit[..., 3]
    x = torch.atan2(sin0, cos0) # range [-pi, pi)
    y = torch.atan2(sin1, cos1) # range [-pi, pi)

    return x, y

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

class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(CircularConv2d(2, 32, 3, 1, 1), nn.SiLU(),
                                 CircularConv2d(32, 64, 5, 1, 2), nn.SiLU(),
                                 CircularConv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 CircularConv2d(64, 32, 5, 1, 2), nn.SiLU(),
                                 CircularConv2d(32, 2, 3, 1, 1))
    def forward(self, x):
        # x shape(B, 2, y, x)
        out = self.seq(x)
        out = out.permute(0, 2, 3, 1)
        return out

class SolNet(nn.Module):
    def __init__(self, flow_net, hidden_dim=256):
        super().__init__()
        
        # Flow prediction net
        self.flow_net = flow_net

        # Flow field encoder: CNN for 64x64x1 input
        self.flow_encoder = nn.Sequential(
            CircularConv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 1, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(1 * 16 * 16, hidden_dim),
            nn.SiLU(),
        )

        # Fully connected layers to fuse flow + position
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4)
        )

    def forward(self, tracer_positions, flow_field):
        """
        tracer_positions: (B, L, 4)
        flow_field: (B, 64, 64, 2)
        Output: (B, L, 4)
        """
        B, L, _ = tracer_positions.shape
        flow_field = flow_field.permute(0, 3, 1, 2)  # (B, 2, 64, 64)

        # Predict flow fields
        flow_next = self.flow_net(flow_field)  # (B, 64, 64, 2)

        # Encode flow fields for tracer prediction
        flow_encoded = self.flow_encoder(flow_field[:,0:1])    # (B, hidden_dim)
        flow_encoded_expanded = flow_encoded.unsqueeze(1).expand(B, L, -1)  # (B, L, hidden_dim)
        fused = torch.cat([flow_encoded_expanded, tracer_positions], dim=-1)  # (B, L, hidden_dim + 4)
        fused_flat = fused.view(B * L, -1)          # (B*L, hidden_dim + 4)
        pos_next_flat = self.predictor(fused_flat)  # (B*L, 4)
        pos_next = pos_next_flat.view(B, L, 4)      # (B, L, 4)

        return pos_next, flow_next

##############################################
################# Train CNN  #################
##############################################
# """
epochs = 500
train_batch_size = 200
val_batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
val_tensor = torch.utils.data.TensorDataset(val_u1[:-1], val_u2[:-1], val_u1[1:], val_u2[1:])
val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=val_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
loss_history = {
    "train_forecast_u1": [],
    "train_forecast_u2": [],
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    }
best_val_loss = float('inf')

flow_net = FlowNet()
solnet = SolNet(flow_net, hidden_dim=256).to(device)
optimizer = torch.optim.Adam(solnet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for ep in range(1, epochs+1):
    solnet.train()
    start_time = time.time()

    train_loss_forecast_u1 = 0.
    train_loss_forecast_u2 = 0.
    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # randomly choosing tracers
        tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
        u1_initial = torch.index_select(u1_initial, dim=1, index=tracer_idx) # (batch, L, 4)
        u1_next    = torch.index_select(u1_next,    dim=1, index=tracer_idx) # (batch, L, 4)

        u1_pred, u2_pred = solnet(u1_initial, u2_initial)
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)
        loss_total = loss_forecast_u1 + loss_forecast_u2

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    loss_history["train_forecast_u1"].append(train_loss_forecast_u1)
    loss_history["train_forecast_u2"].append(train_loss_forecast_u2)
    end_time = time.time()

    # Validation
    if ep % 10 == 0:
        solnet.eval()
        val_loss_u1 = 0.
        val_loss_u2 = 0.
        with torch.no_grad():
            for u1_initial, u2_initial, u1_next, u2_next in val_loader:
                u1_initial, u2_initial, u1_next, u2_next = map(lambda x: x.to(device), [u1_initial, u2_initial, u1_next, u2_next])

                u1_pred, u2_pred = solnet(u1_initial, u2_initial)
                val_loss_u1 += nnF.mse_loss(u1_next, u1_pred).item()
                val_loss_u2 += nnF.mse_loss(u2_next, u2_pred).item()
        val_loss_u1 /= len(val_loader)
        val_loss_u2 /= len(val_loader)
        val_loss_total = val_loss_u1 + val_loss_u2
        loss_history["val_forecast_u1"].append(val_loss_u1)
        loss_history["val_forecast_u2"].append(val_loss_u2)
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            checkpoint = {
                'epoch': ep,
                'model_state_dict': solnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_total
            }
            torch.save(checkpoint, r"../model/CNN_L1024.pt")
            status = "✅"
        else:
            status = ""
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f} "
              f"val_u1: {val_loss_u1:.4f}  val_u2: {val_loss_u2:.4f}  val_total: {val_loss_total:.4f} "
              f"{status}"
              )
    else:
        loss_history["val_forecast_u1"].append(np.nan)
        loss_history["val_forecast_u2"].append(np.nan)
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f} "
              )

np.savez(r"../model/CNN_L1024_loss_history.npz", **loss_history)


# Number of Parameters
cnn_params = parameters_to_vector(solnet.parameters()).numel()
print(f'cnn #parameters:      {cnn_params:,}')
# """
############################################
################# Test CNN #################
############################################
checkpoint = torch.load("../model/CNN_L1024.pt", map_location=device)
solnet.load_state_dict(checkpoint['model_state_dict'])

# # One-step Prediction
# test_batch_size = 200
# test_tensor = torch.utils.data.TensorDataset(test_u)
# test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
# test_u_pred = torch.zeros_like(test_u)
# si = 0
# for u in test_loader:
#     solnet.eval()
#     test_u_batch = u[0].to(device)
#     with torch.no_grad():
#         test_u_pred[si:si+test_batch_size] = solnet(test_u_batch)
#     si += test_batch_size
# # print("MSE (normalized):", nnF.mse_loss(test_u[1:], test_u_pred[:-1]).item())
# # test_u_original_pred = normalizer.decode(test_u_pred)
# test_u_original_pred = test_u_pred
# print("MSE (original):",nnF.mse_loss(test_u2_original[1:], test_u_original_pred[:-1]).item())
# np.save(r"../data/CNN_psi_64x64_OneStepPrediction.npy", test_u_original_pred.to("cpu"))

# One-Step Prediction
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
solnet.eval()
with torch.no_grad():
    test_u1_pred, test_u2_pred = solnet(test_u1, test_u2)
MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
print("MSE1:", MSE1.item())
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")

np.save(r"../data/CNN_L1024_xy_unit_OneStepPrediction.npy", test_u1_pred.to("cpu"))
np.save(r"../data/CNN_L1024_psi_OneStepPrediction.npy", test_u2_pred.to("cpu"))

