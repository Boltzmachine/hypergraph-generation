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

    f = torch.cos(0.5 * torch.pi * ((t / T) + s) / (1 + s))
    alphas = f[1:] / f[:-1]
    betas = 1 - alphas
    assert len(betas) == T
    return betas

@make_registrable()
def linear_beta_schedule(T, beta_1=1e-4, beta_T=0.02):
    betas = torch.linspace(beta_1, beta_T, T)
    return betas

class NoiseScheduler(Model):
    ...

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
        self.register_buffer("Q", self._get_Q(betas))
        self.register_buffer("Q_bar", self._get_Q_bar(betas)) 
        
    def get_Qt(self, t):
        return self.Q[t-1]
    
    def get_Qt_bar(self, t):
        return self.Q_bar[t-1]


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
    

class Transition(Model):
    default_implementation = "default"
    
    def __init__(
        self,
        node_scheduler: Lazy[NoiseScheduler],
        edge_scheduler: Lazy[NoiseScheduler],
        T: int = 1000,
    ):
        super().__init__()
        self.T = T
        self.node_scheduler = node_scheduler.construct(T=T)
        self.edge_scheduler = edge_scheduler.construct(T=T)
        
    def transit(self, X, H, t):
        X = self.transit_X(X, t)
        H = self.transit_H(H, t)
        
        return DataContainer(X=X, H=H)
    
    def transit_X(self, X, t):
        Qt_bar = self.node_scheduler.get_Qt_bar(t)
        X_mn = F.one_hot(X, self.node_scheduler.n_classes).float()
        X_prob = X_mn @ Qt_bar.unsqueeze(1)

        X_prob = X_prob.view(-1, X_prob.size(-1))
        X = X_prob.multinomial(1).view(*X.size())
        
        return X
    
    def transit_H(self, H, t):
        Qt_bar = self.edge_scheduler.get_Qt_bar(t)
        H_mn = F.one_hot(H, self.edge_scheduler.n_classes).float()
        H_prob = H_mn @ Qt_bar.unsqueeze(1)

        H_prob = H_prob.view(-1, H_prob.size(-1))
        H = H_prob.multinomial(1).view(*H.size())
        
        return H
Transition.register("default")(Transition)