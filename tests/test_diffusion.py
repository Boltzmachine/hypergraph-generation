import torch
from src.dataset import DataContainer
from src.diffusion import Diffusion
from src.noise_scheduler import UniformDiscreteNoiseScheduler, Transition
from src.modules.model import DummyModel

from tango.common import Lazy

def test_diffusion():
    K = 2
    T = 100
    X = torch.rand(8, 20, 20)
    H = torch.randint(0, 2, (8, 20, 100))
    
    data = DataContainer(X=X, H=H)
    X, H = data
    assert H.max() < 1.001
    assert H.min() > -0.001

    diffusion = Diffusion(
        model = DummyModel,
        learning_rate = 3e-4,
        transition = Transition(
            node_scheduler=Lazy(UniformDiscreteNoiseScheduler),
            edge_scheduler=Lazy(UniformDiscreteNoiseScheduler),
        )
    )
    t = torch.randint(1, T+1, size=(X.size(0), ))
    X, H_ = diffusion.transit(X, H, t)
    assert torch.abs(H_ - H).sum() <  torch.abs(H_ - (1-H)).sum() 


if __name__ == "__main__":
    test_diffusion()

