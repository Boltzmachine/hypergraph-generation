import torch
from torch import nn
from tango.common.registrable import Registrable


def cosine_beta_schedule_discrete(T, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    t = torch.arange(T+1)

    f = torch.cos(0.5 * torch.pi * ((t / T) + s) / (1 + s))
    alphas = f[1:] / f[:-1]
    betas = 1 - alphas
    assert len(betas) == T
    return betas

class NoiseScheduler(nn.Module, Registrable):
    """
    Registrable class for noise scheduler
    """

NoiseScheduler.register("discrete")
class DiscreteNoiseScheduler(NoiseScheduler):
    def __init__(
            self,
            n_classes: int,
            T: int,
            ) -> None:
        super().__init__()
        self.n_classes = n_classes
        
        betas = cosine_beta_schedule_discrete(T)
        self.register_buffer("Q", self._get_Q(betas))
        self.register_buffer("Q_bar", self._get_Q_bar(betas)) 
        
    def get_Qt(self, t):
        return self.Q[t-1]
    
    def get_Qt_bar(self, t):
        return self.Q_bar[t-1]
    
    def _get_Q(self, betas):     
        """
        Q = (1 - beta) I + beta / K 11^T
        """
        K = self.n_classes
        betas = betas.unsqueeze(1).unsqueeze(1)
        Q =  (1 - betas) * torch.eye(K).unsqueeze(0) + betas / K * torch.ones(K, K).unsqueeze(0)
        return Q
    
    def _get_Q_bar(self, betas):
        """
        Q_bar = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K 11^T
        """
        K = self.n_classes
        cumprods = torch.cumprod(1 - betas, dim=0).unsqueeze(1).unsqueeze(1)
        Q_bar = cumprods * torch.eye(K).unsqueeze(0) + (1 - cumprods) / K * torch.ones(K, K).unsqueeze(0)
        return Q_bar