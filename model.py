
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.15, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return residual + y


class MNIST1020Net(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        in_features = 784 + 28 + 28 + 49 + 6

        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.10)
        )

        self.res1 = ResidualMLPBlock(768, dropout=0.12)
        self.res2 = ResidualMLPBlock(768, dropout=0.12)

        self.down = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.12)
        )

        self.res3 = ResidualMLPBlock(384, dropout=0.15)

        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(192, num_classes)
        )

    @staticmethod
    def _extra_features(x):
        b = x.size(0)
        img = x[:, 0]

        row = img.mean(dim=2)
        col = img.mean(dim=1)

        pooled = img.view(-1, 7, 4, 7, 4).mean(dim=(2, 4))
        pooled = pooled.flatten(start_dim=1)

        mass = img.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        yy, xx = torch.meshgrid(
            torch.arange(28, device=x.device, dtype=x.dtype),
            torch.arange(28, device=x.device, dtype=x.dtype),
            indexing="ij"
        )

        cx = (img * xx).sum(dim=(1, 2), keepdim=True) / mass
        cy = (img * yy).sum(dim=(1, 2), keepdim=True) / mass

        mask = (img > 0.08).float()
        denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)

        x_min = (mask * xx).sum(dim=(1, 2), keepdim=True) / denom
        y_min = (mask * yy).sum(dim=(1, 2), keepdim=True) / denom

        geom = torch.cat([
            cx / 27.0,
            cy / 27.0,
            x_min / 27.0,
            y_min / 27.0,
            img.mean(dim=(1, 2), keepdim=True),
            img.amax(dim=(1, 2), keepdim=True)
        ], dim=2).view(b, 6)

        return torch.cat([row, col, pooled, geom], dim=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        flat = x.flatten(1)
        extra = self._extra_features(x)
        feat = torch.cat([flat, extra], dim=1)

        y = self.in_proj(feat)
        y = self.res1(y)
        y = self.res2(y)
        y = self.down(y)
        y = self.res3(y)

        return self.head(y)
