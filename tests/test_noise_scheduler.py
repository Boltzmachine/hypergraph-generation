import torch
from scipy.stats import normaltest
from matplotlib import pyplot as plt

from src.dataset import CuboidDataset
from src.noise_scheduler import UniformDiscreteNoiseScheduler, GaussianDiscreteNoiseScheduler, cosine_beta_schedule, linear_beta_schedule, GaussianContinuousNoiseScheduler

import pytest

@pytest.mark.parametrize("scheduler_class, beta_schedule", [(UniformDiscreteNoiseScheduler, cosine_beta_schedule), (GaussianDiscreteNoiseScheduler, linear_beta_schedule)])
def test_scheduler_Q_and_Qbar(scheduler_class, beta_schedule):
    T = 50
    K = 2
    scheduler = scheduler_class(beta_schedule, K, T)
    
    assert len(scheduler.Q_bar) == T
    assert len(scheduler.Q) == T
    assert scheduler.Q[0].size() == torch.Size([K, K])
    
    assert torch.isclose(scheduler.Q.sum(-1), torch.ones(T, K)).all()
    assert torch.isclose(scheduler.Q.sum(-2), torch.ones(T, K)).all()
    
    for t in [1, 20, 27, 50]:
        Qi = torch.eye(K)
        for i in range(1, t+1):
            Qi = Qi @ scheduler.get_Qt(i)
        assert torch.isclose(scheduler.get_Qt_bar(t), Qi).all()

@pytest.mark.parametrize("K, scheduler_class, beta_schedule", [(3, UniformDiscreteNoiseScheduler, cosine_beta_schedule), (256, GaussianDiscreteNoiseScheduler, linear_beta_schedule)])   
def test_stationary_uniform(K, scheduler_class, beta_schedule):
    T = 300
    scheduler = scheduler_class(beta_schedule, K, T).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    Q_inf = scheduler.get_Qt_bar(T)
    assert torch.isclose(Q_inf, torch.ones_like(Q_inf) / K, atol=1e-3).all()


def test_stationary_gaussian():
    T = 500
    bs = 10240
    X = torch.stack([CuboidDataset.gen_verts() for _ in range(bs)])
    scheduler = GaussianContinuousNoiseScheduler(linear_beta_schedule, T)
    eps = torch.randn_like(X)
    alpha_bar = scheduler.get_alpha_bar(torch.ones(X.size(0), dtype=torch.long) * T)[:, None, None]
    X = torch.sqrt(alpha_bar) * X + torch.sqrt(1 - alpha_bar) * eps

    true_normal = torch.randn_like(X).view(-1).numpy()
    X = X.view(-1).numpy()
    # plt.hist(X, bins=50)
    # plt.savefig("1.png")

    plt.hist(true_normal, bins=50)
    plt.savefig("2.png")
    

if __name__ == "__main__":
    # test_stationary_uniform(256, GaussianDiscreteNoiseScheduler)
    test_stationary_gaussian()