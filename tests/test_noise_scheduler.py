import torch

from src.noise_scheduler import UniformDiscreteNoiseScheduler, GaussianDiscreteNoiseScheduler

import pytest

def scheduler_Q_and_Qbar(scheduler_class):
    T = 50
    K = 3
    scheduler = scheduler_class(K, T)
    
    assert len(scheduler.Q_bar) == T
    assert len(scheduler.Q) == T
    assert scheduler.Q[0].size() == torch.Size([K, K])
    
    assert torch.isclose(scheduler.Q.sum(-1), torch.ones(T, K)).all()
    assert torch.isclose(scheduler.Q.sum(-2), torch.ones(T, K)).all()
    
    for t in [1, 20, 27, 50]:
        Qi = torch.eye(3)
        for i in range(1, t+1):
            Qi = Qi @ scheduler.get_Qt(i)
        assert torch.isclose(scheduler.get_Qt_bar(t), Qi).all()
    

def test_discrete_noise_scheduler():
    for scheduler_class in [UniformDiscreteNoiseScheduler, GaussianDiscreteNoiseScheduler]:
        scheduler_Q_and_Qbar(scheduler_class)

if __name__ == "__main__":
    scheduler_Q_and_Qbar(UniformDiscreteNoiseScheduler)
    scheduler_Q_and_Qbar(GaussianDiscreteNoiseScheduler)
    