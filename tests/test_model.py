import torch
from src.model import HyperModel, PositionalEncoding

def test_model_equivariant():
    model = HyperModel(128, PositionalEncoding(128, 500))
    model.to(torch.double)
    model.eval()

    bs = 64
    X = torch.randint(0, 256, size=(bs, 8, 3))
    H = torch.randint(0, 2, size=(bs, 6, 8))

    node_perm = torch.stack([torch.randperm(X.size(1)) for _ in range(bs)])
    node_perm_inv = torch.empty_like(node_perm)
    node_perm_inv = torch.scatter(node_perm, dim=1, index=node_perm, src=torch.arange(X.size(1))[None].repeat(bs, 1))
    node_perm = node_perm[..., None].expand(-1, -1, 3)
    node_perm_inv = node_perm_inv[..., None].expand(-1, -1, 3)
  
    assert (X == torch.gather(torch.gather(X, 1, node_perm), 1, node_perm_inv)).all()

    edge_perm = torch.stack([torch.randperm(H.size(1)) for _ in range(bs)])
    edge_perm_inv = torch.empty_like(edge_perm)
    edge_perm_inv = torch.scatter(edge_perm, dim=1, index=edge_perm, src=torch.arange(X.size(1))[None].repeat(bs, 1))
    edge_perm = edge_perm[..., None].expand(-1, -1, 8)
    edge_perm_inv = edge_perm_inv[..., None].expand(-1, -1, 8)

    assert (H == torch.gather(torch.gather(H, 1, edge_perm), 1, edge_perm_inv)).all()

    t = torch.randint(1, 501, size=(bs,))
    X1, H1 = model(X, H, t, torch.ones(bs, 8))

    X1 = torch.gather(X1, 1, node_perm[..., None].expand(-1, -1, -1, 256))
    H1 = torch.gather(torch.gather(H1, 1, edge_perm), 2, node_perm[..., 0][:, None, :].expand(-1, 6, -1))
    X2, H2 = model(
        torch.gather(X, 1, node_perm),
        torch.gather(torch.gather(H, 1, edge_perm), 2, node_perm[..., 0][:, None, :].expand(-1, 6, -1)),
        t,
        torch.ones(bs, 8)
    )

    assert torch.isclose(X1, X2).all()
    assert torch.isclose(H1, H2).all()


def test_model_mask():
    model = HyperModel(128, PositionalEncoding(128, 500))
    model.to(torch.double)
    model.eval()

    bs = 3
    X = torch.randint(0, 256, size=(bs, 7, 3))
    H = torch.randint(0, 2, size=(bs, 10, 7))
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.long)

    t = torch.randint(1, 501, size=(bs,))
    X1, H1 = model(X, H, t, mask)

    X[0, -2, :] = 255 - X[0, -2, :]
    X[2, -1, :] = torch.randint(0, 256, size=(3,))
    H[0, :, -2] = 1 - H[0, :, -2]
    H[2, :, -1] = 1 - H[2, :, -1]
    X2, H2 = model(X, H, t, mask)
    assert torch.isclose(X1 * mask[..., None, None], X2 * mask[..., None, None]).all()
    assert torch.isclose(H1 * mask[:, None, :], H2 * mask[:, None, :]).all()

    X[2, -3, :] = 255 - X[2, -3, :]
    X3, H3 = model(X, H, t, mask)
    assert (~torch.isclose(X1 * mask[..., None, None], X3 * mask[..., None, None])).any()
    assert (~torch.isclose(H1 * mask[:, None, :], H3 * mask[:, None, :])).any()


if __name__ == "__main__":
    test_model_mask()