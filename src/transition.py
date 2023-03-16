import torch
from torch import nn
import torch.nn.functional as F
from tango.common import Lazy, Registrable
from tango.integrations.torch import Model

from .noise_scheduler import *


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
        
        return X, H
    
    def transit_X(self, X, t):
        if self.node_scheduler.name == "identity":
            return (X, torch.zeros_like(X))
        if isinstance(self.node_scheduler, DiscreteNoiseScheduler):
            return self.transit_X_discrete(X, t)
        elif isinstance(self.node_scheduler, ContinuousNoiseScheduler):
            return self.transit_X_continuous(X, t)
        raise ValueError        

    def transit_X_continuous(self, X, t):
        """
        X - [bs, n_nodes, 3]
        """
        eps = torch.randn_like(X)
        alpha_bar = self.node_scheduler.get_alpha_bar(t)[:, None, None]
        X = torch.sqrt(alpha_bar) * X + torch.sqrt(1 - alpha_bar) * eps

        return X, eps

    def transit_X_discrete(self, X, t):
        Qt_bar = self.node_scheduler.get_Qt_bar(t)
        X_mn = F.one_hot(X, self.node_scheduler.n_classes).float()
        X_prob = X_mn @ Qt_bar.unsqueeze(1)

        X_prob = X_prob.view(-1, X_prob.size(-1))
        X = X_prob.multinomial(1).view(*X.size())
        
        return X
    
    def transit_H(self, H, t):
        Qt_bar = self.edge_scheduler.get_Qt_bar(t)
        
        Qt_bar = F.one_hot(Qt_bar.view(-1, Qt_bar.size(-1)).multinomial(1), self.edge_scheduler.n_classes).view(*Qt_bar.size()).float()
        H_mn = F.one_hot(H, self.edge_scheduler.n_classes).float()
        H_prob = H_mn @ Qt_bar.unsqueeze(1)
        import pudb; pu.db
        H = H_prob.argmax(-1)

        # H_prob = H_prob.view(-1, H_prob.size(-1))
        # H = H_prob.multinomial(1).view(*H.size())
        
        return H, Qt_bar
    
    def denoise(self, X: tuple, H: tuple, t, deterministic):
        if self.node_scheduler.name == "identity":
            X = X[1]
        else:
            X = self.denoise_X(X, t, deterministic)
        if self.edge_scheduler.name == "identity":
            H = H[1]
        else:
            H = self.denoise_H(H, t, deterministic)
        return X, H

    def denoise_X(self, X, t, deterministic):
        if isinstance(self.node_scheduler, DiscreteNoiseScheduler):
            return self.denoise_X_discrete(X, t, deterministic)
        elif isinstance(self.node_scheduler, ContinuousNoiseScheduler):
            return self.denoise_X_continuous(X, t, deterministic)
        raise ValueError        

    def denoise_X_continuous(self, X: tuple, t, deterministic):
        """
        X0 is actually eps here
        """
        X0, Xt = X
        z = torch.randn_like(Xt) if deterministic else 0

        alpha = self.node_scheduler.get_alpha(t)[:, None, None]
        alpha_bar = self.node_scheduler.get_alpha_bar(t)[:, None, None]
        beta = self.node_scheduler.get_beta(t)[:, None, None]

        X = 1. / torch.sqrt(alpha) * (Xt - (1-alpha)/(1e-10+torch.sqrt(1 - alpha_bar)) * X0) + torch.sqrt(beta) * z

        return X
            
    def denoise_X_discrete(self, X, t, deterministic):
        raise NotImplementedError
        X_prob = torch.softmax(X, -1)
        if deterministic:
            X = X_prob.argmax(-1)
        else:
            X_prob = X_prob.view(-1, X_prob.size(-1))
            X = X_prob.multinomial(1).view(*X.size()[:-1])
        X = self.transit_X(X, t-1)
        return X

    def denoise_H(self, H, t, deterministic):
        """
        H0 is actually noise here
        """
        # H0, Ht - [bs, n_hyper, n_nodes]
        H0, Ht = H
        Ht = Ht.float()

        H0 = Ht @ H0.transpose(-1, -2)
        bs, n1, n2 = H0.size()
        
        H0 = torch.sigmoid(H0)
        H0 = torch.stack([1-H0, H0], dim=-1)
        Ht = torch.stack([1-Ht, Ht], dim=-1)
        
        H0 = H0.view(bs, n1*n2, 2)
        Ht = Ht.view(bs, n1*n2, 2)
        
        if t[0] > 1.5:
            # xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T

            Qt = self.edge_scheduler.get_Qt(t)
            Qt_bar = self.edge_scheduler.get_Qt_bar(t)
            # s=t-1
            Qs_bar = self.edge_scheduler.get_Qt_bar(t-1)
            
            Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
            left_term = Ht @ Qt_T                      # bs, N, d_t-1
            left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

            right_term = Qs_bar.unsqueeze(1)               # bs, 1, d0, d_t-1
            numerator = left_term * right_term          # bs, N, d0, d_t-1

            Ht_T = Ht.transpose(-1, -2)      # bs, dt, N

            prod = Qt_bar @ Ht_T                 # bs, d0, N
            prod = prod.transpose(-1, -2)               # bs, N, d0
            denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
            denominator[denominator == 0] = 1e-6

            q_s_given_t_and_0 = numerator / denominator # bs, N, d, d
            
            weighted = H0.unsqueeze(-1) * q_s_given_t_and_0         # bs, n, d0, d_t-1
            unnormalized_prob_H = weighted.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_H[torch.sum(unnormalized_prob_H, dim=-1) == 0] = 1e-5
            H_prob = unnormalized_prob_H / torch.sum(unnormalized_prob_H, dim=-1, keepdim=True)  # bs, n, d_t-1
        else:
            H_prob = H0
        
        if deterministic:
            H = H_prob.argmax(-1)
        else:
            H_prob = H_prob.view(-1, H_prob.size(-1))
            H = H_prob.multinomial(1)
        
        H = H.view(bs, n1, n2)
                    
        return H

Transition.register("default")(Transition)