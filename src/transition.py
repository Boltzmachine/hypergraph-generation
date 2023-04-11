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
        
    def transit(self, X, E, H, t):
        X = self.transit_X(X, t)
        E = self.transit_E(E, t)
        H = self.transit_H(H, t)
        
        return X, E, H
    
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
    
    def transit_E(self, E, t):
        if self.edge_scheduler.name == "identity":
            return (E, torch.zeros_like(E))
        E0 = E
        Qt_bar = self.edge_scheduler.get_Qt_bar(t)
        E_mn = F.one_hot(E, self.edge_scheduler.n_classes).float()
    
        E_prob = E_mn @ Qt_bar.unsqueeze(1)

        E_prob = E_prob.view(-1, E_prob.size(-1))
        E = E_prob.multinomial(1).view(*E.size())
        E = torch.triu(E, diagonal=1)
        E = E + E.transpose(-1, -2)
        
        return E, E0
    
    def transit_H(self, H, t):
        if self.edge_scheduler.name == "identity":
            return (H, torch.zeros_like(H))
        if isinstance(self.edge_scheduler, DiscreteNoiseScheduler):
            return self.transit_H_discrete(H, t)
        elif isinstance(self.edge_scheduler, ContinuousNoiseScheduler):
            return self.transit_H_continuous(H, t)
        raise ValueError  
    
    def transit_H_continuous(self, H, t):
        """
        X - [bs, n_nodes, 3]
        """
        eps = torch.randn_like(H.float())
        alpha_bar = self.edge_scheduler.get_alpha_bar(t)[:, None, None]
        H = torch.sqrt(alpha_bar) * H + torch.sqrt(1 - alpha_bar) * eps

        return H, eps

    def transit_H_discrete(self, H, t):
        H0 = H
        Qt_bar = self.edge_scheduler.get_Qt_bar(t)
        H_mn = F.one_hot(H, self.edge_scheduler.n_classes).float()
        H_prob = H_mn @ Qt_bar.unsqueeze(1)

        H_prob = H_prob.view(-1, H_prob.size(-1))
        H = H_prob.multinomial(1).view(*H.size())
        
        H_noise = torch.logical_xor(H0, H)
        
        return H, H0
    
    def denoise(self, X: tuple, E: tuple, H: tuple, t, deterministic, last_step):
        if self.node_scheduler.name == "identity":
            X = X[1]
        else:
            X = self.denoise_X(X, t, deterministic, last_step)
        if self.edge_scheduler.name == "identity":
            H = H[1]
            E = E[1]
        else:
            H = self.denoise_H(H, t, deterministic, last_step)
            E = self.denoise_E(E, t, deterministic, last_step)
        return X, E, H

    def denoise_X(self, X, t, deterministic, last_step):
        if isinstance(self.node_scheduler, DiscreteNoiseScheduler):
            return self.denoise_X_discrete(X, t, deterministic, last_step)
        elif isinstance(self.node_scheduler, ContinuousNoiseScheduler):
            return self.denoise_X_continuous(X, t, deterministic, last_step)
        raise ValueError        

    def denoise_X_continuous(self, X: tuple, t, deterministic, last_step):
        """
        X0 is actually eps here
        """
        X0, Xt = X
        z = 0 if deterministic else torch.randn_like(Xt)

        alpha = self.node_scheduler.get_alpha(t)[:, None, None]
        alpha_bar = self.node_scheduler.get_alpha_bar(t)[:, None, None]
        beta = self.node_scheduler.get_beta(t)[:, None, None]

        X = 1. / torch.sqrt(alpha) * (Xt - (1-alpha)/torch.sqrt(1 - alpha_bar) * X0) + torch.sqrt(beta) * z

        return X
            
    def denoise_X_discrete(self, X, t, deterministic, last_step):
        raise NotImplementedError
        X_prob = torch.softmax(X, -1)
        if deterministic:
            X = X_prob.argmax(-1)
        else:
            X_prob = X_prob.view(-1, X_prob.size(-1))
            X = X_prob.multinomial(1).view(*X.size()[:-1])
        X = self.transit_X(X, t-1)
        return X

    def denoise_H(self, H, t, deterministic, last_step):
        if isinstance(self.edge_scheduler, DiscreteNoiseScheduler):
            return self.denoise_H_discrete(H, t, deterministic, last_step)
        elif isinstance(self.edge_scheduler, ContinuousNoiseScheduler):
            return self.denoise_H_continuous(H, t, deterministic, last_step)
        raise ValueError
    
    def denoise_H_continuous(self, H, t, deterministic, last_step):
        """
        H0 is actually eps here
        """
        H0, Ht = H
        z = torch.randn_like(Ht) if deterministic else 0

        alpha = self.edge_scheduler.get_alpha(t)[:, None, None]
        alpha_bar = self.edge_scheduler.get_alpha_bar(t)[:, None, None]
        beta = self.edge_scheduler.get_beta(t)[:, None, None]

        H = 1. / torch.sqrt(alpha) * (Ht - (1-alpha)/(1e-10+torch.sqrt(1 - alpha_bar)) * H0) + torch.sqrt(beta) * z
        
        if last_step:
            H = (H > 0.5).long()

        return H

    def denoise_H_discrete(self, H, t, deterministic, last_step):
        """
        H0 is actually noise here
        """
        # H0, Ht - [bs, n_hyper, n_nodes]
        H_noise, Ht = H
        bs, n1, n2 = Ht.size()
        
        # assert Ht.dtype == torch.long
        # H0 = torch.logical_xor(Ht, (H_noise > 0).long())
        H0 = torch.sigmoid(H_noise)
        # H0 = (H0 == H_noise) * H_noise + (H0 != H_noise) * (1-H_noise)
        
        H0 = torch.stack([1-H0, H0], dim=-1)
        Ht = torch.stack([1-Ht, Ht], dim=-1).float()
        
        H0 = H0.view(bs, n1*n2, 2)
        Ht = Ht.view(bs, n1*n2, 2)
        if not last_step:
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
    
    def denoise_E(self, E, t, deterministic, last_step):
        if isinstance(self.edge_scheduler, DiscreteNoiseScheduler):
            return self.denoise_E_discrete(E, t, deterministic, last_step)
        elif isinstance(self.edge_scheduler, ContinuousNoiseScheduler):
            return self.denoise_E_continuous(E, t, deterministic, last_step)
        raise ValueError

    def denoise_E_discrete(self, E, t, deterministic, last_step):
        E_noise, Et = E
        bs, n1, n2 = Et.size()
        assert n1 == n2
        
        E0 = torch.softmax(E_noise, dim=-1)
        Et = torch.stack([1-Et, Et], dim=-1).float()
        
        E0 = E0.view(bs, n1*n2, 2)
        Et = Et.view(bs, n1*n2, 2)
        if not last_step:
            # xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
            Qt = self.edge_scheduler.get_Qt(t)
            Qt_bar = self.edge_scheduler.get_Qt_bar(t)
            # s=t-1
            Qs_bar = self.edge_scheduler.get_Qt_bar(t-1)
            
            Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
            left_term = Et @ Qt_T                      # bs, N, d_t-1
            left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

            right_term = Qs_bar.unsqueeze(1)               # bs, 1, d0, d_t-1
            numerator = left_term * right_term          # bs, N, d0, d_t-1

            Et_T = Et.transpose(-1, -2)      # bs, dt, N

            prod = Qt_bar @ Et_T                 # bs, d0, N
            prod = prod.transpose(-1, -2)               # bs, N, d0
            denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
            
            denominator[denominator == 0] = 1e-6
            q_s_given_t_and_0 = numerator / denominator # bs, N, d, d
            
            weighted = E0.unsqueeze(-1) * q_s_given_t_and_0         # bs, n, d0, d_t-1
            unnormalized_prob_E = weighted.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            E_prob = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)  # bs, n, d_t-1
        else:
            E_prob = E0
        
        if deterministic:
            E = E_prob.argmax(-1)
        else:
            E_prob = E_prob.view(-1, E_prob.size(-1))
            E = E_prob.multinomial(1)
        
        E = E.view(bs, n1, n2)
        E = torch.triu(E,  diagonal=1)
        E = E + E.transpose(-1, -2)
                    
        return E

Transition.register("default")(Transition)