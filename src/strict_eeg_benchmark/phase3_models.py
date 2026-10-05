"""Compact single-view neural reference models; each returns logits and a 128-D embedding."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = ("mlp_de", "eegnet", "temporal_cnn", "spectral_encoder", "graph_encoder")
RAW_MODELS = {"eegnet", "temporal_cnn"}
ARCHITECTURE_DESCRIPTIONS = {
    "mlp_de": "310→256→128→3, GELU, dropout 0.25",
    "eegnet": "8 temporal kernels of width 64; 16 depthwise 62-channel spatial filters; 32 separable temporal filters; 128-D embedding",
    "temporal_cnn": "62→64 temporal convolution, three depthwise-separable residual dilated 64-channel blocks (dilations 1,2,4), 128-channel pooled embedding",
    "spectral_encoder": "shared 5-band→32 projection, two 64-channel spatial 1-D convolutions, attention and mean pooling, 128-D embedding",
    "graph_encoder": "static symmetric 4-neighbor normalized graph, two 64-channel graph convolutions, mean and max pooling, 128-D embedding",
}


class MLPDE(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(310, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(x)
        return self.head(embedding), embedding


class EEGNet(nn.Module):
    """EEGNet-style temporal, depthwise spatial, and separable temporal blocks."""

    def __init__(self, dropout: float):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding="same", bias=False), nn.BatchNorm2d(8)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (62, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False),
            nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6, 128), nn.ELU())
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.separable(self.spatial(self.temporal(x.unsqueeze(1))))
        embedding = self.embedding(h)
        return self.head(embedding), embedding


class DilatedResidual(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class TemporalCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(62, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            DilatedResidual(64, 1, dropout), DilatedResidual(64, 2, dropout),
            DilatedResidual(64, 4, dropout), nn.Conv1d(64, 128, 1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(x)
        return self.head(embedding), embedding


class SpectralEncoder(nn.Module):
    """Share a band projection across channels, then learn spatial relations."""

    def __init__(self, dropout: float):
        super().__init__()
        self.band_projection = nn.Sequential(nn.Linear(5, 32), nn.GELU())
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.channel_attention = nn.Conv1d(64, 1, 1)
        self.embedding = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (62, 5):
            raise ValueError("Spectral input must have shape [batch, 62, 5]")
        h = self.channel_mixer(self.band_projection(x).transpose(1, 2))
        weights = self.channel_attention(h).softmax(dim=-1)
        pooled = torch.cat(((h * weights).sum(dim=-1), h.mean(dim=-1)), dim=1)
        embedding = self.embedding(pooled)
        return self.head(embedding), embedding


class GraphEncoder(nn.Module):
    """Two static normalized-adjacency graph-convolution layers on 62 nodes."""

    def __init__(self, adjacency: np.ndarray, dropout: float):
        super().__init__()
        if adjacency.shape != (62, 62):
            raise ValueError("Expected a 62 × 62 adjacency matrix")
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.input_projection = nn.Linear(5, 32)
        self.gcn1 = nn.Linear(32, 64, bias=False)
        self.gcn2 = nn.Linear(64, 64, bias=False)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (62, 5):
            raise ValueError("Graph input must have shape [batch, 62, 5]")
        h = F.gelu(self.input_projection(x))
        h = self.dropout(F.gelu(self.norm1(self.gcn1(torch.matmul(self.adjacency, h)))))
        h = self.dropout(F.gelu(self.norm2(self.gcn2(torch.matmul(self.adjacency, h)))))
        embedding = self.embedding(torch.cat((h.mean(dim=1), h.amax(dim=1)), dim=1))
        return self.head(embedding), embedding


def anatomical_adjacency(channel_names: tuple[str, ...], neighbors: int = 4) -> tuple[np.ndarray, dict]:
    """Static symmetric k-NN graph from standard_1005 electrode positions."""
    import mne

    if len(channel_names) != 62 or neighbors < 1:
        raise ValueError("Expected 62 channels and a positive neighbor count")
    montage = mne.channels.make_standard_montage("standard_1005")
    supplied = {key.upper(): value for key, value in montage.get_positions()["ch_pos"].items()}
    if set(channel_names) - set(supplied) != {"CB1", "CB2"}:
        raise ValueError(f"Unexpected channels missing from standard_1005: {sorted(set(channel_names) - set(supplied))}")
    # SEED's CB1/CB2 are absent from standard_1005. Fix them near the corresponding occipital side.
    supplied["CB1"] = supplied["O1"] + np.array([-0.010, -0.025, -0.015])
    supplied["CB2"] = supplied["O2"] + np.array([0.010, -0.025, -0.015])
    points = np.stack([supplied[name] for name in channel_names])
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    adjacency = np.zeros((62, 62), dtype=np.float32)
    for node in range(62):
        adjacency[node, np.argsort(distance[node])[:neighbors]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(62, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    normalized = adjacency / np.sqrt(degree[:, None] * degree[None, :])
    return normalized.astype(np.float32), {
        "montage": "MNE standard_1005", "neighbors": neighbors, "symmetrized": True,
        "self_loops": True, "channel_names": list(channel_names),
        "CB1_rule": "O1 + [-0.010, -0.025, -0.015] metre",
        "CB2_rule": "O2 + [0.010, -0.025, -0.015] metre",
        "adjacency": normalized.tolist(),
    }


def build_model(name: str, dropout: float, adjacency: np.ndarray | None = None) -> nn.Module:
    if name == "mlp_de":
        return MLPDE(dropout)
    if name == "eegnet":
        return EEGNet(dropout)
    if name == "temporal_cnn":
        return TemporalCNN(dropout)
    if name == "spectral_encoder":
        return SpectralEncoder(dropout)
    if name == "graph_encoder":
        if adjacency is None:
            raise ValueError("Static anatomical adjacency is required for graph encoder")
        return GraphEncoder(adjacency, dropout)
    raise ValueError(f"Unknown Phase-3 model: {name}")
