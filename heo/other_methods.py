import numpy as np
import torch
import torch.nn as nn
import time
from math import pi


def mcge(dim, energy_func, times=10000, lr=1., grad_sample_time=1, momentum=0.9,rep_energy=False):
    p = torch.ones(dim, device='cuda:0') * 0.5  # * 0.5
    grad = torch.zeros([dim], device='cuda:0')
    if rep_energy:
        energy_s = []
        uncern_s = []
    for t in range(times):
        x = torch.sgn(p.unsqueeze(-1) - torch.rand([dim, grad_sample_time], device='cuda:0'))
        energy = energy_func(x)
        grad_new = torch.zeros([dim, grad_sample_time], device='cuda:0')
        grad_new[x > 0] += (energy.unsqueeze(0) / p.unsqueeze(-1)).expand(-1, grad_sample_time)[x > 0]
        grad_new[x < 0] -= ((energy.unsqueeze(0) / (1 - p.unsqueeze(-1)))).expand(-1, grad_sample_time)[x < 0]
        grad_new = grad_new.mean(dim=1)
        grad = grad*momentum + lr * grad_new
        p -= grad
        p.data = torch.clamp(p.data, 0, 1)
        if rep_energy:
            energy_s.append((energy_func(2 * (p.unsqueeze(-1) > 0.5) - 1)).detach().cpu().numpy()[0])
            uncern_s.append((p * (1 - p)).sum().detach().cpu().numpy())
    if rep_energy:
        return p.detach().cpu(), energy_s, uncern_s
    else:
        return p.detach().cpu()

# CIM, SIM-CIM, aSB, bSB, dSB
def isingMac(method, couplings, times=20000, dt=0.1, max_eigenvalue=None, 
             rep_enegry=False, gap=False):
    dim = couplings.shape[0]
    if rep_enegry:
        energy_s = []
    if method in ['CIM','SIM-CIM']:
        x = torch.cos(2 * np.pi * torch.randn(dim, device='cuda:0')) * 0.001   
        a_0 = 1
        c_0 = 1 / max_eigenvalue
    elif method in ['aSB','bSB','dSB']:
        x = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
        y = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
        J = torch.sqrt((couplings * couplings).sum() / (dim * (dim - 1)))
        c_0 = 0.5 / J / np.sqrt(dim)
        a_0 = 1
    a = 0

    for t in range(times):
        if method in ['CIM','SIM-CIM']:
            a += 1.1 * a_0 / times
        elif method in ['aSB','bSB','dSB']:
            a += a_0 / times
        if method == 'CIM':
            x += dt * (-(x ** 2 + a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
        elif method == 'SIM-CIM':
            x += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1)) + 0.25 * torch.randn_like(x)
            x[x > 1] = 1
            x[x < -1] = -1
        elif method == 'aSB':
            y += dt * (-(x ** 2 + 1 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
            x += dt * y
        elif method == 'bSB':
            y += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
            x += dt * y
            mask = (x.abs() > 1)
            x[mask] = torch.sgn(x)[mask]
            y[mask] = 0
        elif method == 'dSB':
            y += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, torch.sgn(x).reshape(-1)))
            x += dt * y
            mask = (x.abs() > 1)
            x[mask] = torch.sgn(x)[mask]
            y[mask] = 0
        if rep_enegry:
            energy_s.append((torch.dot(torch.matmul(couplings, torch.sgn(x)), torch.sgn(x))) / 2)
    if gap:
        final_enegry = (torch.dot(torch.matmul(couplings, torch.sgn(x)), torch.sgn(x))) / 2
        return final_enegry.cpu().numpy(), final_enegry - (torch.dot(torch.matmul(couplings, x), x)) / 2
    else:
        x = torch.sgn(x)
        final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2

        if rep_enegry:
            return energy_s
        else:
            return final_enegry.cpu().numpy()

#LQA
class Lqa_basic():
    def __init__(self, couplings, device='cpu'):
        super(Lqa_basic, self).__init__()
        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1], device=device)
        self.weights = (2 * torch.rand(self.n, device=device) - 1) * 0.1
        self.velocity = torch.zeros(self.n, device=device)
        self.grad = torch.zeros(self.n, device=device)
        self.device = device

    def forward(self, t, step, mom, g):
        # Implements momentum assisted gradient descent update
        w = torch.tanh(self.weights)
        a = 1 - torch.tanh(self.weights) ** 2
        # spin x,z values
        z = torch.sin(w * pi / 2)
        x = torch.cos(w * pi / 2)

        # gradient
        self.grad = ((1 - t) * z + 2 * t * g * torch.matmul(self.couplings, z) * x) * a * pi / 2
        # weight update
        self.velocity = mom * self.velocity - step * self.grad
        self.weights = self.weights + self.velocity

    def schedule(self, i, N):
        return i / N

    def energy_ising(self, config):
        # energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def minimise(self,
                 step=2,  # step size
                 g=1,  # gamma in the article
                 N=200,  # no of iterations
                 mom=0.99,  # momentum
                 f=0.1  # multiplies the weight initialisation
                 ):
        self.weights = (2 * torch.rand(self.n, device=self.device) - 1) * f
        self.velocity = torch.zeros([self.n], device=self.device)
        self.grad = torch.zeros([self.n], device=self.device)

        time0 = time.time()

        for i in range(N):
            t = self.schedule(i, N)
            self.forward(t, step, mom, g)

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights.detach())
        self.energy = float(self.energy_ising(self.config))

        print('min energy ' + str(self.energy))


class Lqa(nn.Module):
    def __init__(self, couplings, device='cpu'):
        super(Lqa, self).__init__()

        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1], device=device)
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, 1], device=device)
        self.weights = torch.zeros([self.n], device=device)
        self.device = device

    def schedule(self, i, N):
        # annealing schedule
        return i / N

    def energy_ising(self, config):
        # ising energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def energy_full(self, t, g):
        # g for gamma in the paper
        # cost function value
        config = torch.tanh(self.weights) * pi / 2
        ez = self.energy_ising(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t * ez * g - (1 - t) * ex)

    def minimise(self,
                 step=1,  # learning rate
                 N=200,  # no of iterations
                 g=1.,
                 f=1.,
                 gap=False):

        self.weights = (2 * torch.rand([self.n], device=self.device) - 1) * f
        self.weights.requires_grad = True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights], lr=step)

        for i in range(N):
            t = self.schedule(i, N)
            energy = self.energy_full(t, g)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights.detach())
        self.energy = float(self.energy_ising(self.config))
        if gap:
            return self.energy, self.energy - float(self.energy_ising(self.weights.detach()))
        else:
            return self.energy

