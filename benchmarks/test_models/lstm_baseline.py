from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn


@dataclass
class LSTMBaselineConfig:
    input_length: int
    forecast_horizon: int
    num_variables: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class LSTMBaseline(nn.Module):
    """
    Variable-wise LSTM encoder with MLP head to forecast horizon.
    Input:  (B, T, M)
    Output: {'forecasts': (B, M, H), 'interpretability': {...}}
    """

    def __init__(self, config: LSTMBaselineConfig):
        super().__init__()
        self.cfg = config
        factor = 2 if config.bidirectional else 1

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size * factor, config.hidden_size * factor),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * factor, config.forecast_horizon),
        )

    @staticmethod
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, M = x.shape
        u = x.permute(0, 2, 1).contiguous().view(B * M, T, 1)
        out, (h_n, _) = self.encoder(u)
        h_last = h_n[-1]
        y = self.head(h_last)  # (B*M, H)
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





