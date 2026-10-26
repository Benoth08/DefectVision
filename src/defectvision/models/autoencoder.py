"""Modele non supervise : autoencodeur entraine uniquement sur des pieces saines.

Principe : l'autoencodeur apprend a reconstruire l'aspect normal d'une piece.
Sur une piece defectueuse, la zone du defaut est mal reconstruite ; l'erreur
de reconstruction donne une carte d'anomalie (localisation) et un score
d'image (decision). Aucune image de defaut n'est vue pendant l'entrainement.

Deux variantes :
- FeatureAutoencoder (par defaut) : reconstruit les caracteristiques
  intermediaires d'un ResNet-18 gele (couches 2 et 3). Il compare des motifs
  plutot que des pixels, ce qui le rend moins sensible au bruit de texture ;
- ConvAutoencoder : reconstruit directement les pixels (perte L2 + SSIM).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Flou gaussien separable sur des cartes (B, 1, H, W)."""
    if sigma <= 0:
        return maps
    k = gaussian_kernel1d(sigma, maps.device, maps.dtype)
    pad = k.numel() // 2
    out = F.conv2d(F.pad(maps, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1))


def ssim_map(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """Carte SSIM locale entre deux lots d'images a un canal dans [0, data_range]."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = gaussian_blur(x, sigma), gaussian_blur(y, sigma)
    sxx = gaussian_blur(x * x, sigma) - mu_x**2
    syy = gaussian_blur(y * y, sigma) - mu_y**2
    sxy = gaussian_blur(x * y, sigma) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return num / den


def reduce_maps(maps: torch.Tensor, reduction: str = "max", topk_ratio: float = 0.01, blur_sigma: float = 4.0):
    """Carte d'anomalie (B, 1, H, W) -> score d'image (B,)."""
    maps = gaussian_blur(maps, blur_sigma)
    flat = maps.flatten(1)
    if reduction == "max":
        return flat.max(dim=1).values
    if reduction == "topk_mean":
        k = max(1, int(round(flat.shape[1] * topk_ratio)))
        return flat.topk(k, dim=1).values.mean(dim=1)
    if reduction == "mean":
        return flat.mean(dim=1)
    raise ValueError(f"reduction inconnue : {reduction}")


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, True),
    )


class ConvAutoencoder(nn.Module):
    """Autoencodeur convolutif sur pixels (1 canal, cote multiple de 32)."""

    tensor_mode = "gray"

    def __init__(self, latent_channels: int = 64, ssim_weight: float = 0.5, **_):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.encoder = nn.Sequential(
            _down(1, 32), _down(32, 64), _down(64, 128), _down(128, 256), _down(256, latent_channels)
        )
        # L'upsampling + convolution evite les artefacts en damier des convolutions transposees.
        self.decoder = nn.Sequential(
            _up(latent_channels, 256),
            _up(256, 128),
            _up(128, 64),
            _up(64, 32),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        l2 = F.mse_loss(recon, x)
        ssim = ssim_map(recon, x).mean()
        return (1 - self.ssim_weight) * l2 + self.ssim_weight * (1 - ssim)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        structural = ((1 - ssim_map(recon, x)) / 2).clamp(0, 1)
        return 0.5 * structural + 0.5 * (recon - x).abs()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class FeatureAutoencoder(nn.Module):
    """Autoencodeur 1x1 sur les caracteristiques ResNet-18 (couches 2 et 3, gelees)."""

    tensor_mode = "rgb"
    feature_channels = 128 + 256

    def __init__(self, pretrained: bool = True, hidden_channels: int = 192, latent_channels: int = 64, **_):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for module in (self.stem, self.layer2, self.layer3):
            for param in module.parameters():
                param.requires_grad = False
        c = self.feature_channels
        self.register_buffer("feat_mean", torch.zeros(1, c, 1, 1))
        self.register_buffer("feat_std", torch.ones(1, c, 1, 1))
        # Moyenne locale 3x3 : chaque position voit un petit voisinage.
        self.context = nn.AvgPool2d(3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(c, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, c, 1),
        )
        # Des la construction, le ResNet doit etre en mode evaluation : sinon le
        # calcul de normalisation modifierait ses statistiques BatchNorm.
        self.train(True)

    def train(self, mode: bool = True):
        # Le ResNet reste en mode evaluation (statistiques BatchNorm d'ImageNet).
        super().train(mode)
        for module in (self.stem, self.layer2, self.layer3):
            module.eval()
        return self

    @torch.no_grad()
    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.context(torch.cat([f2, f3], dim=1))

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return (self.raw_features(x) - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def fit_normalization(self, batches) -> None:
        """Moyenne et ecart-type par canal, calcules sur les images saines d'entrainement."""
        total, total_sq, count = 0.0, 0.0, 0
        for x in batches:
            f = self.raw_features(x)
            total = total + f.sum(dim=(0, 2, 3), keepdim=True)
            total_sq = total_sq + (f**2).sum(dim=(0, 2, 3), keepdim=True)
            count += f.shape[0] * f.shape[2] * f.shape[3]
        mean = total / count
        var = (total_sq / count - mean**2).clamp_min(1e-8)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(var.sqrt())

    def forward(self, x: torch.Tensor):
        f = self.features(x)
        return self.decoder(self.encoder(f)), f

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, f = self(x)
        return F.mse_loss(recon, f)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        recon, f = self(x)
        err = ((recon - f) ** 2).mean(dim=1, keepdim=True)
        return F.interpolate(err, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor):
        return None


ARCHITECTURES = {"conv": ConvAutoencoder, "feature": FeatureAutoencoder}


def build_autoencoder(cfg: dict, pretrained: bool | None = None) -> nn.Module:
    """Construit l'autoencodeur decrit par la section 'autoencoder' de params.yaml."""
    arch = cfg.get("architecture", "feature")
    if arch not in ARCHITECTURES:
        raise ValueError(f"architecture inconnue : {arch}")
    kwargs = {
        "latent_channels": int(cfg.get("latent_channels", 64)),
        "hidden_channels": int(cfg.get("hidden_channels", 192)),
        "ssim_weight": float(cfg.get("ssim_weight", 0.5)),
        "pretrained": bool(cfg.get("pretrained", True)) if pretrained is None else pretrained,
    }
    return ARCHITECTURES[arch](**kwargs)
