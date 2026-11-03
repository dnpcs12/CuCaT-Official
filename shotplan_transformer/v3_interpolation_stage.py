from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class V3InterpolationStageConfig:
    """Configuration for the post shot-plan interpolation classifier."""

    field_vocab_sizes: dict[str, int]
    interpolation_vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.15
    max_frame_delta: int = 240


class V3InterpolationStageModel(nn.Module):
    """Predict segment interpolation from neighboring discrete shot plans.

    The main shot-plan model predicts only semantic camera labels.  This small
    post-stage consumes previous/current/next shot-plan labels and timing deltas
    and predicts the interpolation/easing label for the segment starting at the
    current keyframe.
    """

    def __init__(self, config: V3InterpolationStageConfig) -> None:
        super().__init__()
        self.config = config
        self.fields = tuple(config.field_vocab_sizes.keys())
        dim = int(config.embedding_dim)
        self.field_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(vocab_size) + 1, dim, padding_idx=int(vocab_size))
                for field, vocab_size in config.field_vocab_sizes.items()
            }
        )
        self.position_embedding = nn.Embedding(3, dim)
        self.delta_proj = nn.Linear(4, dim)

        input_dim = dim * 4
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        hidden = int(config.hidden_dim)
        for layer_index in range(max(1, int(config.num_layers))):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
        layers.append(nn.Linear(hidden, int(config.interpolation_vocab_size)))
        self.mlp = nn.Sequential(*layers)

    def _delta_features(self, prev_delta: torch.Tensor, next_delta: torch.Tensor) -> torch.Tensor:
        max_delta = max(1.0, float(self.config.max_frame_delta))
        prev_delta = prev_delta.float()
        next_delta = next_delta.float()
        linear_prev = torch.clamp(prev_delta / max_delta, min=0.0, max=1.0)
        linear_next = torch.clamp(next_delta / max_delta, min=0.0, max=1.0)
        log_prev = torch.log1p(torch.clamp(prev_delta, min=0.0)) / torch.log1p(
            torch.tensor(max_delta, dtype=prev_delta.dtype, device=prev_delta.device)
        )
        log_next = torch.log1p(torch.clamp(next_delta, min=0.0)) / torch.log1p(
            torch.tensor(max_delta, dtype=next_delta.dtype, device=next_delta.device)
        )
        return torch.stack([linear_prev, linear_next, log_prev, log_next], dim=-1)

    def _encode_position(self, labels: dict[str, torch.Tensor], position: str, position_id: int) -> torch.Tensor:
        token = None
        for field, embedding in self.field_embeddings.items():
            ids = labels[f"{position}_{field}"].to(next(embedding.parameters()).device).long()
            field_repr = embedding(ids)
            token = field_repr if token is None else token + field_repr
        assert token is not None
        token = token / max(1, len(self.field_embeddings))
        pos = self.position_embedding(
            torch.full((token.shape[0],), int(position_id), dtype=torch.long, device=token.device)
        )
        return token + pos

    def forward(
        self,
        labels: dict[str, torch.Tensor],
        *,
        prev_delta: torch.Tensor,
        next_delta: torch.Tensor,
    ) -> torch.Tensor:
        prev_token = self._encode_position(labels, "prev", 0)
        current_token = self._encode_position(labels, "current", 1)
        next_token = self._encode_position(labels, "next", 2)
        delta_token = self.delta_proj(self._delta_features(prev_delta.to(current_token.device), next_delta.to(current_token.device)))
        features = torch.cat([prev_token, current_token, next_token, delta_token], dim=-1)
        return self.mlp(features)
