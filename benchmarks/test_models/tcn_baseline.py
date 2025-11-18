from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.kernel_size = kernel_size
        self.dilation = dilation

    def _causal_trim(self, y: torch.Tensor) -> torch.Tensor:
        # Trim right-padding to enforce causality
        trim = (self.kernel_size - 1) * self.dilation
        if trim == 0:
            return y
        return y[..., :-trim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self._causal_trim(y)
        y = self.dropout(y)
        y = F.relu(self.conv2(y))
        y = self._causal_trim(y)
        y = self.dropout(y)
        res = self.downsample(x) if self.downsample is not None else x
        # match length in case of trim
        if res.size(-1) != y.size(-1):
            res = res[..., : y.size(-1)]
        return F.relu(y + res)


@dataclass
class TCNBaselineConfig:
    input_length: int
    forecast_horizon: int
    num_variables: int
    channels: Tuple[int, ...] = (64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1


class TCNBaseline(nn.Module):
    """
    Variable-wise TCN encoder; final state to horizon via linear head.
    Input:  (B, T, M)
    Output: {'forecasts': (B, M, H), 'interpretability': {...}}
    """

    def __init__(self, cfg: TCNBaselineConfig):
        super().__init__()
        self.cfg = cfg
        chs = [1] + list(cfg.channels)
        blocks = []
        for i in range(1, len(chs)):
            dilation = 2 ** (i - 1)
            blocks.append(TemporalBlock(chs[i - 1], chs[i], cfg.kernel_size, dilation, cfg.dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(chs[-1], cfg.forecast_horizon)

    @staticmethod
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, M = x.shape
        u = x.permute(0, 2, 1).contiguous().view(B * M, 1, T)  # (B*M, 1, T)
        z = self.tcn(u)  # (B*M, C, T')
        feat = z[:, :, -1]  # (B*M, C)
        y = self.head(feat)  # (B*M, H)
        forecasts = y.view(B, M, self.cfg.forecast_horizon)
        return {
            "forecasts": forecasts,
            "interpretability": {
                "temporal_attention": None,
                "cross_attention": None,
                "compression_attention": None,
                "spline_parameters": None,
                "variable_embeddings": None,
                "cross_embeddings": None,
                "compressed_repr": None,
            },
        }




