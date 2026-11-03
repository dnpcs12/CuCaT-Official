from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 20000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, *, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.pointwise_in(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x).transpose(1, 2)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1)
        return residual + x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        x_attn, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return residual + self.dropout(x_attn)


class CrossAttentionModule(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardModule(dim, expansion=4, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = query
        q = self.query_norm(query)
        c = self.context_norm(context)
        attn_out, _ = self.attn(
            q,
            c,
            c,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)
        x = x + 0.5 * self.ffn(x)
        x = self.final_norm(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1)
        return x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.self_attn = MultiHeadSelfAttentionModule(dim, num_heads=num_heads, dropout=dropout)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.conv(x, valid_mask=valid_mask)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_norm(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1)
        return x


class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.audio_to_dance = CrossAttentionModule(dim, num_heads=num_heads, dropout=dropout)
        self.dance_to_audio = CrossAttentionModule(dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_context = audio
        dance_context = dance
        audio = self.audio_to_dance(
            audio,
            dance_context,
            key_padding_mask=key_padding_mask,
            valid_mask=valid_mask,
        )
        dance = self.dance_to_audio(
            dance,
            audio_context,
            key_padding_mask=key_padding_mask,
            valid_mask=valid_mask,
        )
        return audio, dance


class MultimodalCutKeyframeConformer(nn.Module):
    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
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
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
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
        self.cut_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
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
        *,
        valid_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if audio_features.ndim != 3 or dance_features.ndim != 3:
            raise ValueError("Expected [batch, frames, dim] inputs for both modalities.")

        key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=audio_features.dtype)
            key_padding_mask = ~(valid_mask.bool())

        audio = self.audio_in(audio_features)
        dance = self.dance_in(dance_features)
        audio = self.positional_encoding(audio)
        dance = self.positional_encoding(dance)

        if valid_mask is not None:
            audio = audio * valid_mask.unsqueeze(-1)
            dance = dance * valid_mask.unsqueeze(-1)

        for block in self.audio_blocks:
            audio = block(audio, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.dance_blocks:
            dance = block(dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.cross_blocks:
            audio, dance = block(audio, dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        fused = torch.cat([audio, dance, audio * dance, torch.abs(audio - dance)], dim=-1)
        fused = self.fusion_proj(fused)
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        for block in self.fusion_blocks:
            fused = block(fused, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        cut_logits = self.cut_head(fused).squeeze(-1)
        keyframe_logits = self.keyframe_head(fused).squeeze(-1)
        outputs = {
            "cut_logits": cut_logits,
            "keyframe_logits": keyframe_logits,
        }
        if return_features:
            outputs["audio_features"] = audio
            outputs["dance_features"] = dance
            outputs["fused_features"] = fused
        return outputs


class FiLMMultimodalCutKeyframeConformer(nn.Module):
    """Conformer cut/keyframe model with dance-conditioned FiLM fusion.

    This keeps the same audio/dance inputs as ``MultimodalCutKeyframeConformer``.
    The FiLM and residual fusion layers are zero-initialized so the model starts
    close to an audio-primary Conformer and learns dance conditioning only when it
    helps the cut/keyframe objective.
    """

    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
        model_dim: int = 192,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        audio_layers: int = 2,
        dance_layers: int = 2,
        cross_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        film_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.film_scale = float(film_scale)
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

        self.film_norm = nn.LayerNorm(model_dim)
        self.film = nn.Linear(model_dim, model_dim * 2)
        self.dance_residual_norm = nn.LayerNorm(model_dim)
        self.dance_residual = nn.Linear(model_dim, model_dim)
        self.concat_norm = nn.LayerNorm(model_dim * 4)
        self.concat_residual = nn.Linear(model_dim * 4, model_dim)
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.fusion_dropout = nn.Dropout(dropout)
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
        self.cut_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.keyframe_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self._reset_fusion_to_audio_primary()

    def _reset_fusion_to_audio_primary(self) -> None:
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.dance_residual.weight)
        nn.init.zeros_(self.dance_residual.bias)
        nn.init.zeros_(self.concat_residual.weight)
        nn.init.zeros_(self.concat_residual.bias)

    def forward(
        self,
        audio_features: torch.Tensor,
        dance_features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if audio_features.ndim != 3 or dance_features.ndim != 3:
            raise ValueError("Expected [batch, frames, dim] inputs for both modalities.")

        key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=audio_features.dtype)
            key_padding_mask = ~(valid_mask.bool())

        audio = self.positional_encoding(self.audio_in(audio_features))
        dance = self.positional_encoding(self.dance_in(dance_features))

        if valid_mask is not None:
            audio = audio * valid_mask.unsqueeze(-1)
            dance = dance * valid_mask.unsqueeze(-1)

        for block in self.audio_blocks:
            audio = block(audio, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.dance_blocks:
            dance = block(dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)
        for block in self.cross_blocks:
            audio, dance = block(audio, dance, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        gamma, beta = self.film(self.film_norm(dance)).chunk(2, dim=-1)
        modulated_audio = audio * (1.0 + self.film_scale * torch.tanh(gamma)) + self.film_scale * beta
        dance_residual = self.dance_residual(self.dance_residual_norm(dance))
        concat_features = torch.cat(
            [
                modulated_audio,
                dance,
                modulated_audio * dance,
                torch.abs(modulated_audio - dance),
            ],
            dim=-1,
        )
        concat_residual = self.concat_residual(self.concat_norm(concat_features))
        fused = self.fusion_norm(modulated_audio + dance_residual + concat_residual)
        fused = self.fusion_dropout(fused)

        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        for block in self.fusion_blocks:
            fused = block(fused, key_padding_mask=key_padding_mask, valid_mask=valid_mask)

        cut_logits = self.cut_head(fused).squeeze(-1)
        keyframe_logits = self.keyframe_head(fused).squeeze(-1)
        outputs = {
            "cut_logits": cut_logits,
            "keyframe_logits": keyframe_logits,
        }
        if return_features:
            outputs["audio_features"] = audio
            outputs["dance_features"] = dance
            outputs["fused_features"] = fused
        return outputs


def _odd_kernel_size(value: int) -> int:
    kernel_size = max(3, int(value))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


class LocalTemporalFusionBlock(nn.Module):
    """Local-only temporal block used for the short context branch."""

    def __init__(self, dim: int, *, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        kernel_size = _odd_kernel_size(kernel_size)
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardModule(dim, expansion=2, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.pointwise_in(y)
        y = self.glu(y)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.pointwise_out(y)
        y = self.dropout(y).transpose(1, 2)
        x = residual + y
        x = x + 0.5 * self.ffn(x)
        x = self.final_norm(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1)
        return x


class LocalTemporalFusionBranch(nn.Module):
    """Builds per-frame features from a bounded local music+dance window."""

    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
        model_dim: int,
        short_window_frames: int,
        short_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.short_window_frames = int(short_window_frames)
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
        self.fusion_in = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                LocalTemporalFusionBlock(
                    model_dim,
                    kernel_size=self.short_window_frames,
                    dropout=dropout,
                )
                for _ in range(max(1, int(short_layers)))
            ]
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        dance_features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio = self.audio_in(audio_features)
        dance = self.dance_in(dance_features)
        if valid_mask is not None:
            audio = audio * valid_mask.unsqueeze(-1)
            dance = dance * valid_mask.unsqueeze(-1)

        fused = torch.cat([audio, dance, audio * dance, torch.abs(audio - dance)], dim=-1)
        fused = self.fusion_in(fused)
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1)
        for block in self.blocks:
            fused = block(fused, valid_mask=valid_mask)
        return fused


class MultiScaleFiLMMultimodalCutKeyframeConformer(nn.Module):
    """FiLM Conformer with parallel long and short temporal context branches.

    The long branch is the existing 8-second FiLM Conformer path. The short
    branch is a local-only temporal convolution path whose kernel is controlled
    by ``short_window_frames``. With the current 30 fps data, the default
    ``short_window_frames=120`` sees roughly 4 seconds around each frame.
    """

    def __init__(
        self,
        *,
        audio_input_dim: int,
        dance_input_dim: int,
        model_dim: int = 192,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        audio_layers: int = 2,
        dance_layers: int = 2,
        cross_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        film_scale: float = 0.5,
        short_window_frames: int = 120,
        short_layers: int = 1,
    ) -> None:
        super().__init__()
        self.short_window_frames = int(short_window_frames)
        self.long_branch = FiLMMultimodalCutKeyframeConformer(
            audio_input_dim=audio_input_dim,
            dance_input_dim=dance_input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_expansion=ff_expansion,
            conv_kernel_size=conv_kernel_size,
            audio_layers=audio_layers,
            dance_layers=dance_layers,
            cross_layers=cross_layers,
            fusion_layers=fusion_layers,
            dropout=dropout,
            film_scale=film_scale,
        )
        self.short_branch = LocalTemporalFusionBranch(
            audio_input_dim=audio_input_dim,
            dance_input_dim=dance_input_dim,
            model_dim=model_dim,
            short_window_frames=short_window_frames,
            short_layers=short_layers,
            dropout=dropout,
        )
        self.scale_fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale_norm = nn.LayerNorm(model_dim)
        self.cut_delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.keyframe_delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self._reset_scale_fusion()

    def _reset_scale_fusion(self) -> None:
        linear = self.scale_fusion[1]
        if isinstance(linear, nn.Linear):
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        for head in (self.cut_delta_head, self.keyframe_delta_head):
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        audio_features: torch.Tensor,
        dance_features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        long_outputs = self.long_branch(
            audio_features,
            dance_features,
            valid_mask=valid_mask,
            return_features=True,
        )
        long_fused = long_outputs["fused_features"]
        short_fused = self.short_branch(
            audio_features,
            dance_features,
            valid_mask=valid_mask,
        )
        scale_features = torch.cat(
            [
                long_fused,
                short_fused,
                long_fused * short_fused,
                torch.abs(long_fused - short_fused),
            ],
            dim=-1,
        )
        fused = self.scale_norm(long_fused + self.scale_fusion(scale_features))
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=audio_features.dtype)
            fused = fused * valid_mask.unsqueeze(-1)

        cut_logits = long_outputs["cut_logits"] + self.cut_delta_head(fused).squeeze(-1)
        keyframe_logits = long_outputs["keyframe_logits"] + self.keyframe_delta_head(fused).squeeze(-1)
        outputs = {
            "cut_logits": cut_logits,
            "keyframe_logits": keyframe_logits,
        }
        if return_features:
            outputs["audio_features"] = long_outputs["audio_features"]
            outputs["dance_features"] = long_outputs["dance_features"]
            outputs["long_fused_features"] = long_fused
            outputs["short_fused_features"] = short_fused
            outputs["fused_features"] = fused
        return outputs
