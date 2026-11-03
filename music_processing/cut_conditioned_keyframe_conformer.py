from __future__ import annotations

import torch
from torch import nn

from music_processing.multimodal_conformer import (
    BidirectionalCrossAttentionBlock,
    ConformerBlock,
    SinusoidalPositionalEncoding,
)


class CutConditionedKeyframeConformer(nn.Module):
    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
        condition_input_dim: int,
        model_dim: int = 192,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        audio_layers: int = 2,
        dance_layers: int = 2,
        cross_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_in = nn.Sequential(
            nn.Linear(audio_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.dance_in = nn.Sequential(
            nn.Linear(dance_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.condition_in = nn.Sequential(
            nn.Linear(condition_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        self.audio_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    model_dim,
                    num_heads=num_heads,
                    ff_expansion=ff_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(audio_layers)
            ]
        )
        self.dance_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    model_dim,
                    num_heads=num_heads,
                    ff_expansion=ff_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(dance_layers)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [BidirectionalCrossAttentionBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(cross_layers)]
        )
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(model_dim * 5),
            nn.Linear(model_dim * 5, model_dim),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    model_dim,
                    num_heads=num_heads,
                    ff_expansion=ff_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(fusion_layers)
            ]
        )
        self.keyframe_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        dance_features: torch.Tensor,
        condition_features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if audio_features.ndim != 3 or dance_features.ndim != 3 or condition_features.ndim != 3:
            raise ValueError("Expected [batch, frames, dim] inputs for audio, dance, and condition.")

        key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=audio_features.dtype)
            key_padding_mask = ~(valid_mask.bool())

        audio = self.audio_in(audio_features)
        dance = self.dance_in(dance_features)
        condition = self.condition_in(condition_features)
        audio = self.positional_encoding(audio)
        dance = self.positional_encoding(dance)
        condition = self.positional_encoding(condition)

        if valid_mask is not None:
            audio = audio * valid_mask.unsqueeze(-1)
            dance = dance * valid_mask.unsqueeze(-1)
            condition = condition * valid_mask.unsqueeze(-1)

        for block in self.audio_blocks:
            audio = block(audio, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.dance_blocks:
            dance = block(dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.cross_blocks:
            audio, dance = block(audio, dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        fused = torch.cat([audio, dance, audio * dance, torch.abs(audio - dance), condition], dim=-1)
        fused = self.fusion_proj(fused)
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        for block in self.fusion_blocks:
            fused = block(fused, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        keyframe_logits = self.keyframe_head(fused).squeeze(-1)
        outputs = {"keyframe_logits": keyframe_logits}
        if return_features:
            outputs["audio_features"] = audio
            outputs["dance_features"] = dance
            outputs["condition_features"] = condition
            outputs["fused_features"] = fused
        return outputs


class DancePrimaryFiLMConcatTransformer(nn.Module):
    """Dance-primary stage-2 keyframe model.

    The dance stream owns the main token representation. Audio is still used in
    two ways: it lightly modulates dance tokens with FiLM, and it is explicitly
    concatenated before a residual fusion projection.
    """

    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
        condition_input_dim: int,
        model_dim: int = 256,
        audio_model_dim: int = 96,
        condition_model_dim: int = 32,
        num_heads: int = 4,
        transformer_layers: int = 4,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        film_scale: float = 0.2,
        fusion_scale: float = 1.0,
        audio_gate_init: float = 0.35,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads: {model_dim} vs {num_heads}")

        audio_gate_init = min(max(float(audio_gate_init), 1e-4), 1.0 - 1e-4)
        self.film_scale = float(film_scale)
        self.fusion_scale = float(fusion_scale)
        self.audio_gate_logit = nn.Parameter(torch.logit(torch.tensor(audio_gate_init, dtype=torch.float32)))

        self.dance_in = nn.Sequential(
            nn.Linear(dance_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_in = nn.Sequential(
            nn.Linear(audio_input_dim, audio_model_dim),
            nn.LayerNorm(audio_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.condition_in = nn.Sequential(
            nn.Linear(condition_input_dim, condition_model_dim),
            nn.LayerNorm(condition_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_to_film = nn.Sequential(
            nn.LayerNorm(audio_model_dim),
            nn.Linear(audio_model_dim, model_dim * 2),
        )
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(model_dim + audio_model_dim + condition_model_dim),
            nn.Linear(model_dim + audio_model_dim + condition_model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * int(ff_expansion),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.final_norm = nn.LayerNorm(model_dim)
        self.keyframe_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        dance_features: torch.Tensor,
        condition_features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if audio_features.ndim != 3 or dance_features.ndim != 3 or condition_features.ndim != 3:
            raise ValueError("Expected [batch, frames, dim] inputs for audio, dance, and condition.")

        key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=audio_features.dtype)
            key_padding_mask = ~(valid_mask.bool())

        dance = self.dance_in(dance_features)
        audio = self.audio_in(audio_features)
        condition = self.condition_in(condition_features)
        audio_gate = torch.sigmoid(self.audio_gate_logit).to(dtype=audio.dtype)
        gated_audio = audio * audio_gate

        gamma, beta = self.audio_to_film(gated_audio).chunk(2, dim=-1)
        dance_film = dance * (1.0 + self.film_scale * torch.tanh(gamma)) + self.film_scale * beta

        fusion_update = self.fusion_proj(torch.cat([dance_film, gated_audio, condition], dim=-1))
        fused = dance_film + self.fusion_scale * fusion_update
        fused = self.positional_encoding(fused)

        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        fused = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        fused = self.final_norm(fused)

        keyframe_logits = self.keyframe_head(fused).squeeze(-1)
        outputs = {"keyframe_logits": keyframe_logits}
        if return_features:
            outputs["audio_features"] = audio
            outputs["gated_audio_features"] = gated_audio
            outputs["dance_features"] = dance
            outputs["dance_film_features"] = dance_film
            outputs["condition_features"] = condition
            outputs["fused_features"] = fused
            outputs["audio_gate"] = audio_gate.detach()
        return outputs
