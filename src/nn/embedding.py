import timm
import torch
import torch.nn.functional as F
from torch import nn


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1, -2)).pow(1.0 / self.p)

    def get_info(self):
        print("GEM Module")


class Normalize(nn.Module):
    def __init__(self, order: int = 2, dim: int = 1):
        super().__init__()
        self._order = order
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self._order, dim=self._dim)

    def get_info(self):
        print("Normalize Module")


class EmbeddingModel(nn.Module):
    def __init__(self, encoder_name: str = 'resnet18', embedding_size: int = 128):
        super().__init__()

        backbone = timm.create_model(encoder_name, pretrained=True, num_classes=0, global_pool='')

        self.network = nn.Sequential(
            backbone,
            Normalize(),
            GeM(),
            nn.Flatten(),
            nn.Linear(in_features=backbone.num_features, out_features=embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_info(self):
        print("Embedding Module")
