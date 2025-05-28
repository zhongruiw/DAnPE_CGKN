import numpy as np
import torch

M_np = np.load('./debug_matrices/M.npy')
M = torch.tensor(M_np, dtype=torch.float32).cuda()
M_inv = torch.inverse(M)
