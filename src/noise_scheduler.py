import torch
from torch import nn
import torch.nn.functional as F
from tango.common import Lazy, Registrable, make_registrable, RegistrableFunction
from tango.integrations.torch import Model

from .dataset import DataContainer


@make_registrable()
def cosine_beta_schedule(T, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    t = torch.arange(T+1)

    f = torch.cos(0.5 * torch.pi * ((t / T) + s) / (1 + s)) ** 2
    alpha = f[1:] / f[:-1]
    beta = 1 - alpha
    assert len(beta) == T
    beta = torch.clamp(beta, max=0.999)
    return beta

@make_registrable()
def linear_beta_schedule(T, beta_1=1e-4, beta_T=0.02):
    beta = torch.linspace(beta_1, beta_T, T)
    return beta

class NoiseScheduler(Model):
    ...


class ContinuousNoiseScheduler(NoiseScheduler):
    ...


@NoiseScheduler.register("identity_continuous")
class IdentityContinuousNoiseScheduler(ContinuousNoiseScheduler):
    name = "identity"
    def __init__(
        self, 
        beta_schedule: RegistrableFunction,
        T: int = 1000) -> None:
        super().__init__()
        self.T = T
        betas = beta_schedule(T)
        alpha = 1 - betas
        alpha_bar = torch.cumprod(alpha, 0)
        
        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        
    def get_alpha(self, t):
        return torch.ones_like(self.alpha[t-1])

    def get_alpha_bar(self, t):
        return torch.ones_like(self.alpha_bar[t-1])
    
    def get_beta(self, t):
        return torch.zeros_like(self.beta[t-1])

@NoiseScheduler.register("gaussian_continuous")
class GaussianContinuousNoiseScheduler(ContinuousNoiseScheduler):
    name = "gaussian"
    def __init__(
            self, 
            beta_schedule: RegistrableFunction,
            T: int = 1000) -> None:
        super().__init__()
        self.T = T
        betas = beta_schedule(T)
        alpha = 1 - betas
        alpha_bar = torch.cumprod(alpha, 0)
        
        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        
    def get_alpha(self, t):
        assert (t > 0.5).all()
        return self.alpha[t-1]

    def get_alpha_bar(self, t):
        assert (t > 0.5).all()
        return self.alpha_bar[t-1]
    
    def get_beta(self, t):
        assert (t > 0.5).all()
        return self.beta[t-1]
        

@NoiseScheduler.register("discrete")
class DiscreteNoiseScheduler(NoiseScheduler):
    def __init__(
            self,
            beta_schedule: RegistrableFunction,
            n_classes: int = 2,
            T: int = 1000,
            ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.T = T

        betas = beta_schedule(T)
        self.register_buffer("beta", betas)
        self.register_buffer("Q", self._get_Q(betas))
        self.register_buffer("Q_bar", self._get_Q_bar(betas)) 
        
    def get_Qt(self, t):
        assert (t > 0.5).all()
        return self.Q[t-1]
    
    def get_Qt_bar(self, t):
        assert (t > 0.5).all()
        return self.Q_bar[t-1]
    
    def get_beta(self, t):
        assert (t > 0.5).all()
        return self.beta[t-1]


@NoiseScheduler.register("identity_discrete")
class IdentityDiscreteNoiseScheduler(DiscreteNoiseScheduler):
    name = "identity"

    def __init__(
            self,
            beta_schedule: RegistrableFunction,
            n_classes: int = 2,
            T: int = 1000,
            ) -> None:
        super().__init__(beta_schedule, n_classes, T)
        self.n_classes = n_classes
        self.T = T

    def _get_Q(self, betas):     
        K = self.n_classes
        return torch.eye(K).repeat(len(betas), 1, 1)
    
    def _get_Q_bar(self, betas):
        K = self.n_classes
        return torch.eye(K).repeat(len(betas), 1, 1)
    
    
@NoiseScheduler.register("uniform_discrete")
class UniformDiscreteNoiseScheduler(DiscreteNoiseScheduler):
    name = "uniform"

    def __init__(
            self,
            beta_schedule: RegistrableFunction,
            n_classes: int = 2,
            T: int = 1000
        ) -> None:
        super().__init__(beta_schedule, n_classes, T)
    
    def _get_Q(self, betas):     
        """
        Q = (1 - beta) I + beta / K 11^T
        """
        K = self.n_classes
        betas = betas.unsqueeze(1).unsqueeze(1)
        Q = (1 - betas) * torch.eye(K).unsqueeze(0) + betas / K * torch.ones(K, K).unsqueeze(0)
        return Q
    
    def _get_Q_bar(self, betas):
        """
        Q_bar = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K 11^T
        """
        K = self.n_classes
        cumprods = torch.cumprod(1 - betas, dim=0).unsqueeze(1).unsqueeze(1)
        Q_bar = cumprods * torch.eye(K).unsqueeze(0) + (1 - cumprods) / K * torch.ones(K, K).unsqueeze(0)
        return Q_bar
    

@NoiseScheduler.register("gaussian_discrete")
class GaussianDiscreteNoiseScheduler(DiscreteNoiseScheduler):
    name = "gaussian"
    
    def __init__(
        self,
        beta_schedule: RegistrableFunction,
        n_classes: int = 2, 
        T: int = 1000
    ) -> None:
        super().__init__(beta_schedule, n_classes, T)
    
    def _get_Q(self, betas):     
        K = self.n_classes
        jm = torch.arange(K).unsqueeze(0).expand(K, -1)
        im = torch.transpose(jm, 0, 1)

        log_Q_norm = torch.log( torch.exp( - (4 * torch.arange(-(K-1), K )**2) / ((K-1)**2 * betas)[:, None] ).sum(1) )
        log_Q = - (4 * (im-jm)**2) / ((K-1)**2 * betas)[:, None, None]
        
        Q = torch.exp( log_Q - log_Q_norm[:, None, None] )
        diag_mask = torch.eye(K, dtype=torch.bool).repeat(Q.size(0), 1, 1)
        
        Q[diag_mask] = 0
        Q[diag_mask] = (1 - Q.sum(-1)).view(-1)

        return Q
    
    def _get_Q_bar(self, betas):
        Q_bar = [self.Q[0]]
        
        for Qt in self.Q[1:]:
            Q_bar.append(
                Q_bar[-1] @ Qt
            )
        Q_bar = torch.stack(Q_bar)
        return Q_bar