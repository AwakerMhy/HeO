"""
Non-linear unconstrained binary optimization
"""

import numpy as np
from heo.opt import heo
from heo.log import Logger
import torch
import os

workdir = 'pbo'
os.makedirs(workdir,exist_ok=True)
logger = Logger(f'{workdir}/log.log','pbo')


def simulated_annealing(enegry_func, dim, times=10000, beta0=1, rep_enegry=False):
    x = torch.randint(0, 2, [dim], device='cuda:0') * 2 - 1
    E = enegry_func(x)
    if rep_enegry:
        enegry_s = []
    for t in range(times):
        beta = beta0 * np.log(1 + t / times)
        v = np.random.randint(dim)
        x_ = x.clone()
        x_[v] *= -1
        E_ = enegry_func(x_)
        Delta_E = E_ - E
        if Delta_E < 0 or torch.exp(-beta * Delta_E) >= np.random.random():
            x = x_
            E = E_
        if t % 10000 == 0:
            print('t: {}, beta: {}, E: {}'.format(t, beta, E.min()))
        if rep_enegry:
            enegry_s.append(E.cpu().numpy())

    if rep_enegry:
        return x.cpu(), E.cpu(), np.array(enegry_s)
    else:
        return x.cpu(), E.cpu()


def mcge(dim, enegry_func, times=10000, lr=1., rep_energy=False, grad_sample_time=1):
    p = torch.ones(dim, device='cuda:0') * 0.5  
    momentum = 0.99
    grad = torch.zeros([dim], device='cuda:0')
    if rep_energy:
        energy_s = []
        uncern_s = []
    for t in range(times):
        x = torch.sgn(p.unsqueeze(-1) - torch.rand([dim, grad_sample_time], device='cuda:0'))
        energy = enegry_func(x)
        grad_new = torch.zeros([dim, grad_sample_time], device='cuda:0')
        grad_new[x > 0] += (energy.unsqueeze(0) / p.unsqueeze(-1)).expand(-1, grad_sample_time)[x > 0]
        grad_new[x < 0] -= ((energy.unsqueeze(0) / (1 - p.unsqueeze(-1)))).expand(-1, grad_sample_time)[x < 0]
        grad_new = grad_new.mean(dim=1)
        grad = grad*momentum + lr * grad_new
        p -= grad
        p.data = torch.clamp(p.data, 0, 1)
        if rep_energy:
            energy_s.append((enegry_func(2 * (p.unsqueeze(-1) > 0.5) - 1)).detach().cpu().numpy()[0])
            uncern_s.append((p * (1 - p)).sum().detach().cpu().numpy())
        if t % 1000 == 0:
            print(t, enegry_func(2 * (p.unsqueeze(-1) > 0.5) - 1), ((p * (1 - p)).sum()))
    if rep_energy:
        return p.detach().cpu(), energy_s, uncern_s
    else:
        return p.detach().cpu()


def Hopf(enegry_func, dim, times, lr=2, rep_enegry=False):
    x = torch.rand(dim, device='cuda:0') * 2 - 1
    y = torch.sgn(x)
    if rep_enegry:
        enegry_s = []
        uncern_s = []
    for t in range(times):
        y.requires_grad = True
        energy = enegry_func(y)
        energy.backward()
        x.data = x.data - lr * y.grad
        y.requires_grad = False
        y = torch.sgn(x)
        if rep_enegry:
            enegry_s.append(energy.detach().cpu().numpy())
            uncern_s.append(((1 - x * x) / 4).sum().detach().cpu().numpy())
    if rep_enegry:
        return energy.detach().cpu(), enegry_s, uncern_s
    else:
        return energy.detach().cpu()

dim = 10000
dim2 = 10000

times_sa = 50 
ensemble = 10
beta0 = 10

grad_sample_time = 10

torch.manual_seed(42)

a = 2 * torch.rand([dim]) - 1
b = 2 * torch.rand([dim2]) - 1
W = 2 * torch.rand([dim2, dim]) - 1

a = a.cuda()
b = b.cuda()
W = W.cuda()

energy_func = lambda x: (b * (torch.sigmoid(torch.matmul(W, x + a))).reshape(-1)).sum()
energy_func_ensemble = lambda x: (b.reshape(-1, 1) * (torch.sigmoid(torch.matmul(W, x + a.reshape(-1, 1))))).sum(dim=0)

p = mcge(dim, energy_func_ensemble, times=50000, lr=1e-6, rep_energy=False, grad_sample_time=grad_sample_time)
result = 2 * (p > 0.5).float() - 1
energy = energy_func(result.cuda()).cpu().numpy()
logger.log('mcge energy: {}'.format(energy))


config = {'device':'cuda:0',
          'test_times':10,
           'datapath': r'/example_data/qubo/rudy_all',
           'unbounded':False,
           'type':'sgd',
           'lr':10,
           'momentum':0.9999,
           'T':5000
           }
logger.log(config)

solver =  heo(dim=dim, energy_func=energy_func, opt_config=config)
p = solver.solver(T=config['T'])
result = 2 * (p > 0.5).float() - 1
energy = energy_func(result.cuda()).cpu().numpy()

logger.log('heo energy: {}'.format(energy))

