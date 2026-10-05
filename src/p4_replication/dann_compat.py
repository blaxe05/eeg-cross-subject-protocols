"""32-channel constructor repair for pinned LibEER DannDgcnn.

The released DannDgcnn calls DGCNN.__init__ with no arguments, creating a
62-electrode, three-class graph regardless of the requested dimensions. This
shim initializes the identical pinned components at the requested dimensions; it does not
change the forward graph, loss, or domain adaptation rule.
"""
from __future__ import annotations

import torch


def make_dann(n_channels: int, n_classes: int):
    from models.DGCNN import DGCNN
    from models.DannDgcnn import DannDgcnn, Discriminator

    if n_channels == 62 and n_classes == 3:
        return DannDgcnn(62, 5, n_classes, num_sources=2), "pinned_unmodified_62_channel"
    if n_channels not in (32, 62):
        raise ValueError(n_channels)

    class CompatibleDann(DannDgcnn):
        def __init__(self):
            DGCNN.__init__(self, n_channels, 5, n_classes)
            self.alpha = 0.1
            self.num_sources = 2
            self.discriminator = Discriminator(n_channels * self.layers[-1], 256, 2)
            self.leaky_relus = [torch.nn.LeakyReLU() for _ in self.layers]

    return CompatibleDann(), f"pinned_components_{n_channels}_channel_{n_classes}_class_constructor_repair"
