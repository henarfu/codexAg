import torch
import deepinv as dinv
import numpy as np


import torch
import math
from tqdm import tqdm
# import numpy as np  # only if you need np elsewhere
import math
import torch



class MultipleSNPN_Matrix_PGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter, gamma, eps=1e-6):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter
        self.gamma = gamma
        self.eps = eps

    def forward(self,x0, y, ys, physics_h, physics_s,**kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        eps = self.eps
        fid_old = -np.inf
        idx = 0
        W = torch.eye(ys[0].shape[1]).to(x0.device)

        with torch.no_grad():
            for _ in range(self.max_iter):

                # grad_s = 0
                # for i in range(len(ys)):
                #     grad_s += self.data_fidelity.grad(x_k, ys[i], physics_s[i])*self.gamma[i]
                
                u = x_k - ( self.stepsize * self.data_fidelity.grad(
                    x_k, y, physics_h) + self.gamma[idx]*self.data_fidelity.grad(x_k, ys[idx], physics_s[idx])) # Gradient step
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                fid_i = self.data_fidelity(x_k, y, physics_h).mean().item() +  self.data_fidelity(x_k, ys[idx], physics_s[idx]).mean().item()
                res = np.abs((fid_i - fid_old)/(fid_old + eps))
                if res < eps and idx < len(ys) - 1:
                    idx += 1
                    print(f"Switching to sensor {idx+1} at iteration {_}")
                fid_old = fid_i
                xks.append(x_k)
        return x_k,xks

    # def compute_s_gradients(self, ys, physics_s,i):



class MultipleSNPN_PGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter, gamma, eps=1e-6):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter
        self.gamma = gamma
        self.eps = eps

    def forward(self,x0, y, ys, physics_h, physics_s,type_select='seq',**kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        eps = self.eps
        fid_old = -np.inf
        idx = 0
        with torch.no_grad():
            for _ in range(self.max_iter):
                if type_select == 'concat':
                    grad_s = 0
                    for i in range(len(ys)):
                        grad_s += self.data_fidelity.grad(x_k, ys[i], physics_s[i])*self.gamma[i]
                elif type_select == 'seq':
                    grad_s = self.data_fidelity.grad(x_k, ys[idx], physics_s[idx])*self.gamma[idx]*10
                elif type_select == 'rand':
                    rand_idx = np.random.randint(0,len(ys))
                    grad_s = self.data_fidelity.grad(x_k, ys[rand_idx], physics_s[rand_idx])*self.gamma[rand_idx]*10

                u = x_k - ( self.stepsize * self.data_fidelity.grad(
                    x_k, y, physics_h) + grad_s) # Gradient step
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                fid_i = self.data_fidelity(x_k, y, physics_h).mean().item() +  self.data_fidelity(x_k, ys[idx], physics_s[idx]).mean().item()
                res = np.abs((fid_i - fid_old)/(fid_old + eps))
                if res < eps and idx < len(ys) - 1:
                    idx += 1
                    print(f"Switching to sensor {idx+1} at iteration {_}")
                fid_old = fid_i
                xks.append(x_k)
        return x_k,xks

    # def compute_s_gradients(self, ys, physics_s,i):




class NPN_PGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter, gamma):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter
        self.gamma = gamma

    def forward(self, x0,y, ys, physics_h, physics_s,**kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        with torch.no_grad():
            for _ in range(self.max_iter):
                u = x_k - self.stepsize * ( self.data_fidelity.grad(
                    x_k, y, physics_h) + self.gamma * self.data_fidelity.grad(x_k,ys,physics_s)) # Gradient step
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                xks.append(x_k)
        return x_k,xks
    

class MyPGD(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd, max_iter):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd
        self.max_iter = max_iter

    def forward(self, x0, y, physics, **kwargs):
        """Algorithm forward pass.

        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        x_k = x0

        # Disable autodifferentiation, remove this if you want to unfold
        xks = []
        with torch.no_grad():
            for _ in range(self.max_iter):
                u = x_k - self.stepsize * self.data_fidelity.grad(
                    x_k, y, physics
                )  # Gradient step
                x_k = self.prior.prox(
                    u, sigma_denoiser=self.lambd * self.stepsize
                )  # Proximal step
                xks.append(x_k)
        return x_k,xks
    
