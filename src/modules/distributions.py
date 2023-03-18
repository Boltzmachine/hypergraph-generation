import torch
import numpy as np

from tango.common import Registrable

class Distribution(Registrable):
    ...

@Distribution.register("hist")
class HistDistribution(Distribution):
    def __init__(self, data):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        n_nodes = data['n_nodes']
        prob, bins = np.histogram(n_nodes, bins=np.arange(0, max(n_nodes)+1) + 0.5, density=True)
        prob = torch.tensor(prob)
        self.max_faces = max(data['n_faces'])
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,)) + 1
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p