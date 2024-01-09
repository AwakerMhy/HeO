"""
ternary parameter learning
"""

import numpy as np
import torch

def m_v(mu, sigma):
    return torch.erf(mu / sigma / np.sqrt(2))
def heo_ternary(dim, times=10000, lr=1.):
    p1 = torch.rand([dim2, dim], device='cuda:0')
    p2 = torch.rand([dim2, dim], device='cuda:0')
    p1.requires_grad = True
    p2.requires_grad = True
    for t in range(times):
        sigma = (1 - t / times)
        W = (m_v(p1 - torch.rand([dim2, dim], device='cuda:0'), sigma) +
             m_v(p2 - torch.rand([dim2, dim], device='cuda:0'), sigma)) / 2
        energy = ((new_work_ensemble(W, x) - y_real) ** 2).mean()
        energy.backward()
        p1.data = p1.data - lr * p1.grad
        p1.data = torch.clamp(p1.data, 0, 1)
        p2.data = p2.data - lr * p2.grad
        p2.data = torch.clamp(p2.data, 0, 1)
    return p1.detach().cpu(), p2.detach().cpu()

dim = 100
dim2 = 100
times = 10
rep_enegry = False
batch_size_s = [10, 20, 50, 100, 200, 500, 1000, 2000]
acc_mean_s = []
acc_std_s = []
for batch_size in batch_size_s:
    acc_s = []
    torch.manual_seed(42)
    for _ in range(times):
        print(f'batch size: {batch_size}')
        W_real = torch.randint(-1, 2, [dim2, dim]).float()
        W_real = W_real.cuda()

        new_work_ensemble = lambda W, x: torch.relu(torch.matmul(W, x))
        new_work_real_ensemble = lambda x: torch.relu(torch.matmul(W_real, x))

        x = (torch.rand([dim, batch_size]).cuda() > 0.5).float() + (torch.rand([dim, batch_size]).cuda() > 0.5).float() - 1
        y_real = new_work_real_ensemble(x)
        correction = False
        energy_s = []
        p1, p2 = heo_ternary(dim, times=10000, lr=0.5)
        result_W = (p1 > 0.5).float() + (p2 > 0.5).float() - 1
        acc = ((result_W.cuda() == W_real).float()).mean().cpu().numpy()
        print(f'acc weight: {acc}')
        acc_s.append(acc)
    acc_mean_s.append(np.mean(acc_s))
    acc_std_s.append(np.std(acc_s))
np.savez(r'three_value_acc', acc_mean_s=np.array(acc_mean_s), acc_std_s=np.array(acc_std_s), batch_size_s=batch_size_s)
