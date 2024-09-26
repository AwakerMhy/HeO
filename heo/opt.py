import numpy as np
import torch
import torch.optim as optim


class heo():
    def __init__(self, dim, energy_func, opt_config, device='cuda:0'):
        self.dim = dim
        self.energy_func = energy_func
        self.device = device
        self.opt_config = opt_config

    def phi(self, mu, sigma):
        return torch.erf(mu / sigma)
    
    def initialize(self):
        self.theta = torch.ones(self.dim, device=self.device) * 0.5
        self.theta.requires_grad = True
        x = self.phi(self.theta - torch.rand(self.dim, device=self.device), 1)
        self.energy = self.energy_func(x)
    
    def update(self, t, T):
        sigma = (1 - t / T) * np.sqrt(2)
        x = self.phi(self.theta - torch.rand(self.dim, device=self.device), sigma)
        self.energy = self.energy_func(x)
        self.energy.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.theta.data = torch.clamp(self.theta.data, 0, 1)

    def solver(self, T, save_enegry=False):
        self.initialize()
        if self.opt_config['type'] == 'sgd':
            self.opt=optim.SGD([self.theta], lr=self.opt_config['lr'], momentum=self.opt_config['momentum'])
        elif self.opt_config['type'] == 'adam':
            self.opt=optim.Adam([self.theta], lr=self.opt_config['lr'])
        if save_enegry:
            enegry_s = [self.energy.detach().cpu()]
        for t in range(T):
            self.update(t,T)
            if save_enegry:
                enegry_s.append(self.energy.detach().cpu())
        if save_enegry:
            return self.theta.detach().cpu(), enegry_s
        else:
            return self.theta.detach().cpu()
        
    def __init__(self, dim, energy_func, opt_config, device='cuda:0'):
        self.dim = dim
        self.energy_func = energy_func
        self.device = device
        self.opt_config = opt_config

    def phi(self, mu, sigma):
        return torch.erf(mu / sigma / np.sqrt(2))
    
    def rho(self, x):
        return torch.tan(np.pi*x-0.5*np.pi)
    
    def initialize(self):
        self.theta = torch.ones(self.dim, device=self.device) * 0.5
        self.theta.requires_grad = True
        x = self.phi(self.theta - torch.rand(self.dim, device=self.device), 1)
        self.energy = self.energy_func(x)
    
    def update(self, t, T):
        sigma = (1 - t / T)
        x = self.phi(self.rho(self.theta) - self.rho(torch.rand(self.dim, device=self.device)), sigma)
        self.energy = self.energy_func(x)
        self.energy.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.theta.data = torch.clamp(self.theta.data, 0, 1)

    def solver(self, T, save_enegry=False):
        self.initialize()
        if self.opt_config['type'] == 'sgd':
            self.opt=optim.SGD([self.theta], lr=self.opt_config['lr'], momentum=self.opt_config['momentum'])
        elif self.opt_config['type'] == 'adam':
            self.opt=optim.Adam([self.theta], lr=self.opt_config['lr'])
        if save_enegry:
            enegry_s = [self.energy.detach().cpu()]
        for t in range(T):
            self.update(t,T)
            if save_enegry:
                enegry_s.append(self.energy.detach().cpu())
        if save_enegry:
            return self.theta.detach().cpu(), enegry_s
        else:
            return self.theta.detach().cpu()