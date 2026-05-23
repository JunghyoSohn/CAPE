import numpy as np
import torch
import torch.nn as nn

from ordering.diffan import DiffANOrdering


class LearnableTimeEncoding(nn.Module):
    """SciNO time encoding (matches SciNO-main)."""

    def __init__(self, F=32, M=32, H=1024, D=10, gamma=5.0):
        super().__init__()
        self.F = F
        self.M = M
        self.H = H
        self.D = D
        self.gamma = gamma
        self.Wr = nn.Parameter(torch.randn(F, D) * (gamma ** -1))
        self.mlp = nn.Sequential(
            nn.Linear(2 * F, M),
            nn.GELU(),
            nn.Linear(M, H),
        )

    def forward(self, t):
        projected = torch.matmul(t, self.Wr.T)
        fourier_feature = (1 / np.sqrt(2 * self.F)) * torch.cat(
            [torch.cos(projected), torch.sin(projected)], dim=-1
        )
        return self.mlp(fourier_feature)


class SciNOBackbone(nn.Module):
    """SciNO backbone following SciNO-main architecture."""

    def __init__(self, n_nodes, n_fourier_layers=1, norm_type="batch", gamma=5.0, bias=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_fourier_layers = n_fourier_layers
        self.norm_type = norm_type.lower()
        self.bias = bias

        self.H = max(1024, 5 * self.n_nodes)
        self.S = max(128, 3 * self.n_nodes)

        self.time_encoding = LearnableTimeEncoding(F=32, M=32, H=self.H, D=self.n_nodes, gamma=gamma)

        self.layer1 = nn.Sequential(
            nn.Linear(self.n_nodes, self.H, bias=self.bias),
            nn.LeakyReLU(),
            nn.LayerNorm(self.H),
            nn.Dropout(0.2),
        )

        self.fourier_layers = nn.ModuleList()
        for _ in range(self.n_fourier_layers):
            norm_layer = self._build_norm_layer(2 * self.H)
            self.fourier_layers.append(
                nn.Sequential(
                    nn.Linear(2 * self.H, 2 * self.H, bias=self.bias),
                    nn.LeakyReLU(),
                    norm_layer,
                )
            )

        self.layer2 = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.LeakyReLU(),
            nn.Linear(self.H, self.S),
            nn.LeakyReLU(),
            nn.Linear(self.S, self.n_nodes),
        )

    def _build_norm_layer(self, dim):
        if self.norm_type == "batch":
            return nn.BatchNorm1d(dim)
        if self.norm_type == "layer":
            return nn.LayerNorm(dim)
        raise ValueError("norm_type must be 'batch' or 'layer'")

    def forward(self, x, t):
        t = t.unsqueeze(1).expand(-1, self.n_nodes)
        temb = self.time_encoding(t)
        X = self.layer1(x)
        for layer in self.fourier_layers:
            X_skip = X
            X_fft = torch.fft.fftn(X, s=(X.shape[-1],), norm="ortho", dim=1)
            X_fft = X_fft * temb
            X_fft_real = X_fft.real.flatten(start_dim=1)
            X_fft_imag = X_fft.imag.flatten(start_dim=1)
            X_fft_combined = torch.cat([X_fft_real, X_fft_imag], dim=1)
            X_fft_combined[:, 0::2] = X_fft_real
            X_fft_combined[:, 1::2] = X_fft_imag
            X_t = layer(X_fft_combined)
            X_t_real, X_t_imag = X_t[:, ::2], X_t[:, 1::2]
            X_t_complex = torch.complex(X_t_real, X_t_imag)
            X_ifft = torch.fft.ifftn(X_t_complex, dim=-1, norm="ortho").real
            X = X_ifft + X_skip
        return self.layer2(X)


class SciNOOrdering(DiffANOrdering):
    def __init__(
        self,
        n_nodes,
        masking=True,
        residue=True,
        epochs=3000,
        batch_size=1024,
        learning_rate=0.001,
        cutoff=0.001,
        n_fourier_layers=1,
        norm_type="batch",
        gamma=5.0,
        bias=False,
        n_votes=3,
        early_stopping_wait=300,
        eval_batch_size=None,
    ):
        super().__init__(
            n_nodes=n_nodes,
            masking=masking,
            residue=residue,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            cutoff=cutoff,
            n_votes=n_votes,
            early_stopping_wait=early_stopping_wait,
        )
        self.model = SciNOBackbone(
            n_nodes,
            n_fourier_layers=n_fourier_layers,
            norm_type=norm_type,
            gamma=gamma,
            bias=bias,
        ).to(self.device)
        self.model.float()
        self.opt = torch.optim.AdamW(self.model.parameters(), learning_rate, weight_decay=0.01)


__all__ = ["SciNOOrdering"]
