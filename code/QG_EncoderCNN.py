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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 # nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
        # self.seq = nn.Sequential(nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #                          nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))
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
                                 # nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1))
        # self.seq = nn.Sequential(nn.ConvTranspose2d(1, 16, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        #                          nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        #                          nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1))
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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(32, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 64, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(64, 32, 5, 1, 2), nn.SiLU(),
                                 nn.Conv2d(32, 1, 3, 1, 1))
    def forward(self, x):
        # x shape(t, y, x, 1)
        x = x.permute(0, 3, 1, 2)
        out = self.seq(x)
        out = out.permute(0, 2, 3, 1)
        return out

########################################################
################# Train CNN+AE (Stage1)  ###############
########################################################
dim_u2 = 64*64*2
dim_z = 16*16

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
solnet = CNN().to(device)
params = list(autoencoder.parameters()) + list(solnet.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# """
for ep in range(1, epochs+1):
    solnet.train()
    autoencoder.train()
    start_time = time.time()

    train_loss_forecast_u2 = 0.    
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u2_initial, u2_next in train_loader:
        u2_initial, u2_next = u2_initial.to(device), u2_next.to(device)

        # AutoEncoder
        z_initial = autoencoder.encoder(u2_initial)
        u2_initial_ae = autoencoder.decoder(z_initial)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)

        #  State Forecast
        z_input = z_initial.unsqueeze(-1)  # shape: (B, H, W, 1)
        z_pred = solnet(z_input)
        z_pred = z_pred.squeeze(-1)
        u2_pred = autoencoder.decoder(z_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)

        z_next = autoencoder.encoder(u2_next)
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

torch.save({
    'autoencoder': autoencoder.state_dict(),
    'cnn': solnet.state_dict()
}, "../model/QG_64x64_EncoderCNN.pt")
np.save(r"../model/QG_64x64_EncoderCNN_train_loss_forecast_u2_history_stage1.npy", train_loss_forecast_u2_history)
np.save(r"../model/QG_64x64_EncoderCNN_train_loss_ae_history_stage1.npy", train_loss_ae_history)
np.save(r"../model/QG_64x64_EncoderCNN_train_loss_forecast_z_history_stage1.npy", train_loss_forecast_z_history)

# """
# Load model
checkpoint = torch.load("../model/QG_64x64_EncoderCNN.pt")
autoencoder.load_state_dict(checkpoint['autoencoder'])
solnet.load_state_dict(checkpoint['cnn'])

# Set models to evaluation mode
autoencoder.eval()
solnet.eval()

# Move data to device
test_u2 = test_u2.to(device)

# Inference
with torch.no_grad():
    # Encode full sequence
    test_z = autoencoder.encoder(test_u2)                      # (Ntest, d1, d2)
    test_z_input = test_z.unsqueeze(-1)                        # (Ntest, d1, d2, 1)
    test_z_pred = solnet(test_z_input)                         # (Ntest, d1, d2, 1)
    test_u2_pred = autoencoder.decoder(test_z_pred.squeeze(-1))  # (Ntest, H, W, 2)

# Evaluate one-step MSE
MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
print("MSE2:", MSE2.item())

np.save(r"../data/EncoderCNN_psi_64x64_OneStepPrediction.npy", test_u2_pred.to("cpu"))

# CGKN: Number of Parameters
cnn_params = parameters_to_vector(solnet.parameters()).numel()
encoder_params = parameters_to_vector(autoencoder.encoder.parameters()).numel()
decoder_params = parameters_to_vector(autoencoder.decoder.parameters()).numel()
total_params = cnn_params + encoder_params + decoder_params
print(f'cnn #parameters:      {cnn_params:,}')
print(f'encoder #parameters:  {encoder_params:,}')
print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')