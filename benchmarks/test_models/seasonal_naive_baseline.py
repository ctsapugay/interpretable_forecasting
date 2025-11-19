from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn


@dataclass
class SeasonalNaiveConfig:
    input_length: int
    forecast_horizon: int
    num_variables: int
    season_length: int = 24


class SeasonalNaive(nn.Module):
    """
    Seasonal naive baseline that repeats the last seasonal pattern.
    Input:  (B, T, M)
    Output: {'forecasts': (B, M, H), 'interpretability': {...}}
    """

    def __init__(self, cfg: SeasonalNaiveConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def count_parameters(module: nn.Module) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, M = x.shape
        S = min(self.cfg.season_length, T)
        pattern = x[:, -S:, :]  # (B, S, M)
        pattern_mh = pattern.permute(0, 2, 1).contiguous()  # (B, M, S)
        reps = (self.cfg.forecast_horizon + S - 1) // S
        tiled = pattern_mh.repeat(1, 1, reps)  # (B, M, >=H)
        forecasts = tiled[:, :, : self.cfg.forecast_horizon].contiguous()
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





