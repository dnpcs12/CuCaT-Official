"""ReverseV3 discrete shot-plan model with precomputed cut timing context.

This is the corrected architecture for the "make cuts first, then predict
keyframes" setup:

- Cut timing is assumed to be known before shot-plan prediction.
- A cut is represented by two adjacent keyframes ``t-1, t``; only ``t`` is the
  cut-transition shot-plan target.
- Stage ``cut`` predicts the ReverseV3 shot plan for cut-transition frames.
  Only the previous cut shot-plan may be visible; ordinary keyframe history and
  future cut/keyframe shot-plan labels are masked, while timing positions may
  remain visible.
- Stage ``keyframe`` predicts ordinary keyframe shot plans using nearest
  previous/next cut shot-plan context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


V3Stage = Literal["cut", "keyframe"]


@dataclass
class V3CutContextConfig:
    audio_dim: int
    dance_dim: int
    vocab_sizes: dict[str, int]
    window_size: int
    architecture: str = "transformer"
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 512
    mlp_extra_layers: int = 0
    stage_specific_heads: bool = False
    dropout: float = 0.15
    input_dropout: float = 0.10
    max_position_tokens: int = 16
    max_frame_delta: int = 240
    fusion: str = "film"
    joint_vocab_sizes: dict[str, int] | None = None


class V3CutContextShotPlanModel(nn.Module):
    """Predict ReverseV3 categorical shot-plan fields with cut context."""

    def __init__(self, config: V3CutContextConfig) -> None:
        super().__init__()
        self.config = config
        self.fields = tuple(config.vocab_sizes.keys())
        model_dim = int(config.model_dim)
        self.audio_proj = nn.Linear(int(config.audio_dim), model_dim)
        self.dance_proj = nn.Linear(int(config.dance_dim), model_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, int(config.window_size), model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.input_dropout = nn.Dropout(float(config.input_dropout))
        self.stage_embedding = nn.Embedding(2, model_dim)  # 0=cut, 1=keyframe
        self.context_type_embedding = nn.Embedding(5, model_dim)
        # 0=previous cut, 1=next cut, 2=key timing, 3=cut timing, 4=history
        self.context_visible_embedding = nn.Embedding(2, model_dim)
        self.delta_proj = nn.Linear(2, model_dim)
        self.missing_label_embedding = nn.Parameter(torch.zeros(1, model_dim))
        self.context_position_embedding = nn.Parameter(
            torch.zeros(1, int(config.max_position_tokens), model_dim)
        )

        if config.fusion not in {"film", "dance_film", "concat"}:
            raise ValueError(
                "V3CutContextShotPlanModel supports fusion='film', 'dance_film', or 'concat'."
            )
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.dance_film = None
        self.dance_residual = None
        self.audio_film = None
        self.audio_residual = None
        self.concat_fusion = None
        if config.fusion == "film":
            self.dance_film = nn.Linear(model_dim, model_dim * 2)
            self.dance_residual = nn.Linear(model_dim, model_dim)
        elif config.fusion == "dance_film":
            self.audio_film = nn.Linear(model_dim, model_dim * 2)
            self.audio_residual = nn.Linear(model_dim, model_dim)
        else:
            self.concat_fusion = nn.Sequential(
                nn.LayerNorm(model_dim * 2),
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Dropout(float(config.input_dropout)),
                nn.Linear(model_dim, model_dim),
            )

        self.field_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(vocab_size) + 1, model_dim, padding_idx=int(vocab_size))
                for field, vocab_size in config.vocab_sizes.items()
            }
        )
        self.stage_specific_heads = bool(config.stage_specific_heads)
        if self.stage_specific_heads:
            self.cut_field_heads = nn.ModuleDict(
                {
                    field: nn.Linear(model_dim, int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )
            self.keyframe_field_heads = nn.ModuleDict(
                {
                    field: nn.Linear(model_dim, int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )
            self.cut_joint_heads = nn.ModuleDict(
                {
                    name: nn.Linear(model_dim, int(vocab_size))
                    for name, vocab_size in (config.joint_vocab_sizes or {}).items()
                }
            )
            self.keyframe_joint_heads = nn.ModuleDict(
                {
                    name: nn.Linear(model_dim, int(vocab_size))
                    for name, vocab_size in (config.joint_vocab_sizes or {}).items()
                }
            )
        else:
            self.field_heads = nn.ModuleDict(
                {
                    field: nn.Linear(model_dim, int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )
            self.joint_heads = nn.ModuleDict(
                {
                    name: nn.Linear(model_dim, int(vocab_size))
                    for name, vocab_size in (config.joint_vocab_sizes or {}).items()
                }
            )

        self.architecture = str(config.architecture)
        if self.architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=int(config.num_heads),
                dim_feedforward=int(config.ffn_dim),
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.num_layers))
        elif self.architecture in ("transformer_decoder", "transformer_decoder_full"):
            self.encoder = None
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=int(config.num_heads),
                dim_feedforward=int(config.ffn_dim),
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(config.num_layers))
        elif self.architecture == "mlp":
            self.encoder = None
            mlp_layers: list[nn.Module] = [
                nn.LayerNorm(model_dim * 4),
                nn.Linear(model_dim * 4, int(config.ffn_dim)),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(int(config.ffn_dim), model_dim),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
            ]
            for _ in range(max(0, int(config.mlp_extra_layers))):
                mlp_layers.extend(
                    [
                        nn.Linear(model_dim, model_dim),
                        nn.GELU(),
                        nn.Dropout(float(config.dropout)),
                    ]
                )
            mlp_layers.append(nn.Linear(model_dim, model_dim))
            self.mlp_backbone = nn.Sequential(*mlp_layers)
        else:
            raise ValueError(f"unknown architecture: {self.architecture!r}")
        self.out_norm = nn.LayerNorm(model_dim)
        self.head_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(model_dim, model_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.context_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.missing_label_embedding, std=0.02)
        if self.dance_film is not None:
            nn.init.zeros_(self.dance_film.weight)
            nn.init.zeros_(self.dance_film.bias)
        if self.dance_residual is not None:
            nn.init.zeros_(self.dance_residual.weight)
            nn.init.zeros_(self.dance_residual.bias)
        if self.audio_film is not None:
            nn.init.zeros_(self.audio_film.weight)
            nn.init.zeros_(self.audio_film.bias)
        if self.audio_residual is not None:
            nn.init.zeros_(self.audio_residual.weight)
            nn.init.zeros_(self.audio_residual.bias)

    def _delta_features(self, delta: torch.Tensor) -> torch.Tensor:
        max_delta = max(1.0, float(self.config.max_frame_delta))
        delta = delta.float().reshape(delta.shape[0], -1)
        linear = torch.clamp(delta / max_delta, min=-1.0, max=1.0)
        log = torch.sign(delta) * torch.log1p(torch.abs(delta)) / torch.log1p(
            torch.tensor(max_delta, device=delta.device, dtype=delta.dtype)
        )
        return torch.stack([linear, log], dim=-1)

    def _label_token(
        self,
        labels: dict[str, torch.Tensor] | None,
        mask: torch.Tensor,
        delta: torch.Tensor,
        *,
        context_type: int,
    ) -> torch.Tensor:
        batch_size = mask.shape[0]
        device = mask.device
        visible = (mask.reshape(batch_size) > 0.5).long()
        label_repr = torch.zeros(batch_size, int(self.config.model_dim), device=device)
        for field, embedding in self.field_embeddings.items():
            pad_id = int(self.config.vocab_sizes[field])
            if labels is None or field not in labels:
                field_ids = torch.full((batch_size,), pad_id, device=device, dtype=torch.long)
            else:
                field_ids = labels[field].to(device).reshape(batch_size).long()
                field_ids = torch.where(field_ids < 0, torch.full_like(field_ids, pad_id), field_ids)
                field_ids = torch.where(visible.bool(), field_ids, torch.full_like(field_ids, pad_id))
            label_repr = label_repr + embedding(field_ids)
        label_repr = label_repr / max(1, len(self.field_embeddings))
        label_repr = label_repr + self.missing_label_embedding.reshape(1, -1) * (1.0 - visible.float()).unsqueeze(-1)
        delta_repr = self.delta_proj(self._delta_features(delta).squeeze(1))
        type_repr = self.context_type_embedding(
            torch.full((batch_size,), int(context_type), device=device, dtype=torch.long)
        )
        visible_repr = self.context_visible_embedding(visible)
        return (label_repr + delta_repr + type_repr + visible_repr).unsqueeze(1)

    def _position_tokens(
        self,
        delta: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        context_type: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        count = int(self.config.max_position_tokens)
        if delta is None:
            delta = torch.zeros(batch_size, count, device=device)
        else:
            delta = delta.to(device).float()
        if mask is None:
            mask = torch.zeros(batch_size, count, device=device)
        else:
            mask = mask.to(device).float()
        if delta.shape[1] < count:
            pad = count - delta.shape[1]
            delta = torch.cat([delta, torch.zeros(batch_size, pad, device=device)], dim=1)
            mask = torch.cat([mask, torch.zeros(batch_size, pad, device=device)], dim=1)
        delta = delta[:, :count]
        mask = mask[:, :count]
        tokens = self.delta_proj(self._delta_features(delta))
        tokens = tokens + self.context_position_embedding[:, :count, :]
        tokens = tokens + self.context_type_embedding(
            torch.full((batch_size, count), int(context_type), device=device, dtype=torch.long)
        )
        return tokens * mask.unsqueeze(-1)

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        stage: V3Stage,
        prev_cut_labels: dict[str, torch.Tensor] | None = None,
        prev_cut_mask: torch.Tensor | None = None,
        prev_cut_delta: torch.Tensor | None = None,
        next_cut_labels: dict[str, torch.Tensor] | None = None,
        next_cut_mask: torch.Tensor | None = None,
        next_cut_delta: torch.Tensor | None = None,
        key_position_delta: torch.Tensor | None = None,
        key_position_mask: torch.Tensor | None = None,
        cut_position_delta: torch.Tensor | None = None,
        cut_position_mask: torch.Tensor | None = None,
        history_labels: dict[str, torch.Tensor] | None = None,
        history_mask: torch.Tensor | None = None,
        history_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if stage not in ("cut", "keyframe"):
            raise ValueError(f"stage must be 'cut' or 'keyframe', got {stage!r}")
        batch_size, window_size, _ = audio.shape
        device = audio.device
        audio_tokens = self.audio_proj(audio)
        dance_tokens = self.dance_proj(dance)
        if self.config.fusion == "film":
            assert self.dance_film is not None
            assert self.dance_residual is not None
            gamma, beta = self.dance_film(self.fusion_norm(dance_tokens)).chunk(2, dim=-1)
            fused = audio_tokens * (1.0 + 0.5 * torch.tanh(gamma)) + beta
            fused = fused + self.dance_residual(dance_tokens)
        elif self.config.fusion == "dance_film":
            assert self.audio_film is not None
            assert self.audio_residual is not None
            gamma, beta = self.audio_film(self.fusion_norm(audio_tokens)).chunk(2, dim=-1)
            fused = dance_tokens * (1.0 + 0.5 * torch.tanh(gamma)) + beta
            fused = fused + self.audio_residual(audio_tokens)
        else:
            assert self.concat_fusion is not None
            fused = self.concat_fusion(torch.cat([audio_tokens, dance_tokens], dim=-1))
        fused = fused + self.time_embedding[:, :window_size, :]
        fused = self.input_dropout(fused)

        zero_mask = torch.zeros(batch_size, device=device)
        zero_delta = torch.zeros(batch_size, device=device)
        if prev_cut_mask is None:
            prev_cut_mask = zero_mask
        if prev_cut_delta is None:
            prev_cut_delta = zero_delta
        if next_cut_mask is None:
            next_cut_mask = zero_mask
        if next_cut_delta is None:
            next_cut_delta = zero_delta
        if stage == "cut":
            # Future cut shot-plan values are not known while generating cut
            # shot plans. Timing tokens may remain visible separately.
            next_cut_mask = torch.zeros_like(next_cut_mask)
            if history_mask is not None:
                # Cut generation is the first stage. It may use the previous
                # cut pose, but it should not peek at ordinary keyframe labels.
                history_mask = torch.zeros_like(history_mask)

        prev_cut = self._label_token(
            prev_cut_labels,
            prev_cut_mask,
            prev_cut_delta,
            context_type=0,
        )
        next_cut = self._label_token(
            next_cut_labels,
            next_cut_mask,
            next_cut_delta,
            context_type=1,
        )
        key_pos = self._position_tokens(
            key_position_delta,
            key_position_mask,
            context_type=2,
            batch_size=batch_size,
            device=device,
        )
        cut_pos = self._position_tokens(
            cut_position_delta,
            cut_position_mask,
            context_type=3,
            batch_size=batch_size,
            device=device,
        )

        context_tokens = [prev_cut, next_cut, key_pos, cut_pos]
        if history_labels is not None and history_mask is not None and history_delta is not None:
            # History labels are previous keyframe shot plans only, never future.
            hist_count = history_mask.shape[1]
            hist_tokens = []
            for idx in range(hist_count):
                labels = {field: value[:, idx] for field, value in history_labels.items()}
                hist_tokens.append(
                    self._label_token(
                        labels,
                        history_mask[:, idx],
                        history_delta[:, idx],
                        context_type=4,
                    )
                )
            context_tokens.append(torch.cat(hist_tokens, dim=1))

        cls = self.cls_token.expand(batch_size, -1, -1)
        stage_id = 0 if stage == "cut" else 1
        cls = cls + self.stage_embedding(
            torch.full((batch_size,), stage_id, device=device, dtype=torch.long)
        ).unsqueeze(1)
        if self.architecture == "transformer":
            encoded = self.encoder(torch.cat([cls, *context_tokens, fused], dim=1))
            repr_ = self.out_norm(encoded[:, 0])
        elif self.architecture == "transformer_decoder":
            # Decoder variant: use the stage-aware CLS token as the prediction
            # query, and let it cross-attend over explicit context + feature
            # memory. This keeps the task single-step while testing whether
            # decoder-style query-to-memory attention is preferable to the
            # encoder-only joint self-attention baseline.
            memory = torch.cat([*context_tokens, fused], dim=1)
            decoded = self.decoder(tgt=cls, memory=memory)
            repr_ = self.out_norm(decoded[:, 0])
        elif self.architecture == "transformer_decoder_full":
            # Full-input decoder variant: feed the same full token sequence used
            # by the encoder baseline as decoder tgt, and also expose that same
            # sequence as memory. This is intentionally close to self-attention
            # behavior while still exercising TransformerDecoder layers.
            sequence = torch.cat([cls, *context_tokens, fused], dim=1)
            decoded = self.decoder(tgt=sequence, memory=sequence)
            repr_ = self.out_norm(decoded[:, 0])
        else:
            context = torch.cat(context_tokens, dim=1)
            center_idx = max(0, min(window_size - 1, window_size // 2))
            repr_ = self.mlp_backbone(
                torch.cat(
                    [
                        cls[:, 0],
                        context.mean(dim=1),
                        fused.mean(dim=1),
                        fused[:, center_idx],
                    ],
                    dim=-1,
                )
            )
            repr_ = self.out_norm(repr_)
        repr_ = repr_ + self.head_ffn(repr_)
        if self.stage_specific_heads:
            heads = self.cut_field_heads if stage == "cut" else self.keyframe_field_heads
            joint_heads = self.cut_joint_heads if stage == "cut" else self.keyframe_joint_heads
        else:
            heads = self.field_heads
            joint_heads = self.joint_heads
        outputs = {f"{field}_logits": head(repr_) for field, head in heads.items()}
        for name, head in joint_heads.items():
            outputs[f"{name}_logits"] = head(repr_)
        return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = V3CutContextConfig(
        audio_dim=29,
        dance_dim=168,
        window_size=49,
        vocab_sizes={
            "target_anchor": 7,
            "distance": 7,
            "yaw": 8,
            "pitch": 4,
            "composition": 3,
            "roll": 7,
            "fov": 3,
        },
        model_dim=64,
        num_heads=4,
        num_layers=1,
        ffn_dim=128,
    )
    model = V3CutContextShotPlanModel(cfg)
    batch = 2
    labels = {field: torch.zeros(batch, dtype=torch.long) for field in cfg.vocab_sizes}
    out = model(
        torch.randn(batch, 49, 29),
        torch.randn(batch, 49, 168),
        stage="cut",
        prev_cut_labels=labels,
        prev_cut_mask=torch.ones(batch),
        prev_cut_delta=torch.ones(batch) * 30,
        next_cut_labels=labels,
        next_cut_mask=torch.ones(batch),
        next_cut_delta=torch.ones(batch) * 40,
        key_position_delta=torch.zeros(batch, 16),
        key_position_mask=torch.zeros(batch, 16),
        cut_position_delta=torch.zeros(batch, 16),
        cut_position_mask=torch.zeros(batch, 16),
    )
    assert set(out) == {f"{field}_logits" for field in cfg.vocab_sizes}
    assert out["yaw_logits"].shape == (batch, 8)
    print("V3CutContextShotPlanModel smoke test passed.")
