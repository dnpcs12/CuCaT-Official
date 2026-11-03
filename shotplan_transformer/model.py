from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ShotPlanTransformerConfig:
    audio_dim: int
    dance_dim: int
    max_slots: int
    vocab_sizes: dict[str, int]
    window_size: int = 9
    prev_context_size: int = 2
    future_context_size: int = 0
    model_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    input_dropout: float = 0.0
    fusion: str = "film"
    use_context_tokens: bool = True
    joint_vocab_sizes: dict[str, int] | None = None


class DiscreteShotPlanTransformer(nn.Module):
    def __init__(self, config: ShotPlanTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.max_slots = int(config.max_slots)
        self.fusion = str(config.fusion)
        self.use_context_tokens = bool(config.use_context_tokens)
        self.future_context_size = int(config.future_context_size)

        self.audio_proj = nn.Linear(int(config.audio_dim), int(config.model_dim))
        self.dance_proj = nn.Linear(int(config.dance_dim), int(config.model_dim))
        self.input_dropout = nn.Dropout(float(config.input_dropout))
        self.time_embedding = nn.Parameter(torch.zeros(1, int(config.window_size), int(config.model_dim)))
        if self.use_context_tokens:
            self.context_type_embedding = nn.Parameter(torch.zeros(1, 3, int(config.model_dim)))
        if self.fusion == "interleave":
            self.modality_embedding = nn.Parameter(torch.zeros(1, 2, int(config.model_dim)))
        elif self.fusion == "film":
            self.fusion_norm = nn.LayerNorm(int(config.model_dim))
            self.dance_film = nn.Linear(int(config.model_dim), int(config.model_dim) * 2)
            self.dance_residual = nn.Linear(int(config.model_dim), int(config.model_dim))
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.model_dim)))
        if self.use_context_tokens:
            self.prev_position_embedding = nn.Parameter(
                torch.zeros(1, int(config.prev_context_size), 1, int(config.model_dim))
            )
            self.prev_slot_embedding = nn.Parameter(torch.zeros(1, 1, int(config.max_slots), int(config.model_dim)))
            self.cut_flag_proj = nn.Linear(1, int(config.model_dim))
            self.prev_delta_proj = nn.Linear(1, int(config.model_dim))
            if self.future_context_size > 0:
                self.future_position_embedding = nn.Parameter(
                    torch.zeros(1, self.future_context_size, int(config.model_dim))
                )
                self.future_delta_proj = nn.Linear(1, int(config.model_dim))
            self.prev_field_embeddings = nn.ModuleDict(
                {
                    field: nn.Embedding(int(vocab_size) + 1, int(config.model_dim), padding_idx=int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.model_dim),
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ffn_dim),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.num_encoder_layers))

        self.cls_norm = nn.LayerNorm(int(config.model_dim))
        self.slot_queries = nn.Parameter(torch.zeros(1, int(config.max_slots), int(config.model_dim)))
        self.slot_attention = nn.MultiheadAttention(
            embed_dim=int(config.model_dim),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(int(config.model_dim))
        self.slot_ffn = nn.Sequential(
            nn.Linear(int(config.model_dim), int(config.model_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.model_dim) * 2, int(config.model_dim)),
        )

        self.count_head = nn.Linear(int(config.model_dim), int(config.max_slots))
        self.field_heads = nn.ModuleDict(
            {
                field: nn.Linear(int(config.model_dim), int(vocab_size))
                for field, vocab_size in config.vocab_sizes.items()
            }
        )
        self.joint_heads = nn.ModuleDict(
            {
                name: nn.Linear(int(config.model_dim), int(vocab_size))
                for name, vocab_size in (config.joint_vocab_sizes or {}).items()
            }
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        if self.use_context_tokens:
            nn.init.trunc_normal_(self.context_type_embedding, std=0.02)
        if self.fusion == "interleave":
            nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        if self.use_context_tokens:
            nn.init.trunc_normal_(self.prev_position_embedding, std=0.02)
            nn.init.trunc_normal_(self.prev_slot_embedding, std=0.02)
            if self.future_context_size > 0:
                nn.init.trunc_normal_(self.future_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        if self.fusion == "film":
            nn.init.zeros_(self.dance_film.weight)
            nn.init.zeros_(self.dance_film.bias)
            nn.init.zeros_(self.dance_residual.weight)
            nn.init.zeros_(self.dance_residual.bias)

    def _build_prev_tokens(
        self,
        prev_labels: dict[str, torch.Tensor] | None,
        prev_slot_mask: torch.Tensor | None,
        prev_frame_delta: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        prev_count = int(self.config.prev_context_size)
        if prev_labels is None or prev_slot_mask is None:
            return self.context_type_embedding[:, 1:2, :].expand(batch_size, prev_count, -1)

        prev_slot_mask = prev_slot_mask.to(device)
        label_repr = torch.zeros(
            batch_size,
            prev_count,
            self.max_slots,
            int(self.config.model_dim),
            device=device,
        )
        for field, embedding in self.prev_field_embeddings.items():
            labels = prev_labels[field].to(device)
            pad_id = int(self.config.vocab_sizes[field])
            labels = torch.where(labels < 0, torch.full_like(labels, pad_id), labels)
            label_repr = label_repr + embedding(labels)
        label_repr = label_repr / max(1, len(self.prev_field_embeddings))
        label_repr = label_repr + self.prev_position_embedding[:, :prev_count, :, :]
        label_repr = label_repr + self.prev_slot_embedding[:, :, : self.max_slots, :]

        mask = prev_slot_mask.unsqueeze(-1)
        summed = (label_repr * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp_min(1.0)
        prev_tokens = summed / denom

        if prev_frame_delta is not None:
            delta = torch.log1p(prev_frame_delta.to(device).clamp_min(0.0)).unsqueeze(-1) / 5.0
            prev_tokens = prev_tokens + self.prev_delta_proj(delta)
        prev_tokens = prev_tokens + self.context_type_embedding[:, 1:2, :]
        return prev_tokens

    def _build_future_tokens(
        self,
        future_frame_delta: torch.Tensor | None,
        future_frame_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        future_count = int(self.config.future_context_size)
        if future_count <= 0:
            return torch.zeros(batch_size, 0, int(self.config.model_dim), device=device)
        if future_frame_delta is None:
            delta = torch.zeros(batch_size, future_count, device=device)
        else:
            delta = torch.log1p(future_frame_delta.to(device).clamp_min(0.0)) / 5.0
        if future_frame_mask is None:
            mask = torch.zeros(batch_size, future_count, device=device)
        else:
            mask = future_frame_mask.to(device)
        future_tokens = self.future_delta_proj(delta.unsqueeze(-1))
        future_tokens = future_tokens + self.future_position_embedding[:, :future_count, :]
        future_tokens = future_tokens + self.context_type_embedding[:, 2:3, :]
        return future_tokens * mask.unsqueeze(-1)

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        prev_labels: dict[str, torch.Tensor] | None = None,
        prev_slot_mask: torch.Tensor | None = None,
        prev_frame_delta: torch.Tensor | None = None,
        future_frame_delta: torch.Tensor | None = None,
        future_frame_mask: torch.Tensor | None = None,
        cut_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, window_size, _ = audio.shape
        device = audio.device

        audio_tokens = self.audio_proj(audio)
        dance_tokens = self.dance_proj(dance)
        time_embed = self.time_embedding[:, :window_size, :]
        if self.fusion == "interleave":
            audio_tokens = audio_tokens + time_embed + self.modality_embedding[:, 0:1, :]
            dance_tokens = dance_tokens + time_embed + self.modality_embedding[:, 1:2, :]
            stacked = torch.stack((audio_tokens, dance_tokens), dim=2)
            fused_tokens = stacked.reshape(batch_size, window_size * 2, self.config.model_dim)
        else:
            gamma, beta = self.dance_film(self.fusion_norm(dance_tokens)).chunk(2, dim=-1)
            fused_tokens = audio_tokens * (1.0 + 0.5 * torch.tanh(gamma)) + beta
            fused_tokens = fused_tokens + self.dance_residual(dance_tokens) + time_embed
        fused_tokens = self.input_dropout(fused_tokens)

        cls = self.cls_token.expand(batch_size, -1, -1)
        if self.use_context_tokens:
            if cut_flag is None:
                cut_flag = torch.zeros(batch_size, device=device)
            cut_token = self.cut_flag_proj(cut_flag.to(device).reshape(batch_size, 1, 1))
            cut_token = cut_token + self.context_type_embedding[:, 0:1, :]
            prev_tokens = self._build_prev_tokens(
                prev_labels,
                prev_slot_mask,
                prev_frame_delta,
                batch_size,
                device,
            )
            future_tokens = self._build_future_tokens(
                future_frame_delta,
                future_frame_mask,
                batch_size,
                device,
            )
            encoder_input = torch.cat([cls, cut_token, prev_tokens, future_tokens, fused_tokens], dim=1)
        else:
            encoder_input = torch.cat([cls, fused_tokens], dim=1)
        encoded = self.encoder(encoder_input)
        cls_repr = self.cls_norm(encoded[:, 0])
        memory = encoded[:, 1:]

        slot_queries = self.slot_queries.expand(batch_size, -1, -1)
        slot_repr, _ = self.slot_attention(slot_queries, memory, memory, need_weights=False)
        slot_repr = self.slot_norm(slot_repr + cls_repr.unsqueeze(1))
        slot_repr = slot_repr + self.slot_ffn(slot_repr)

        outputs = {"count_logits": self.count_head(cls_repr)}
        for field, head in self.field_heads.items():
            outputs[f"{field}_logits"] = head(slot_repr)
        for name, head in self.joint_heads.items():
            outputs[f"{name}_logits"] = head(slot_repr)
        return outputs


class _CrossModalFusionBlock(nn.Module):
    def __init__(self, config: ShotPlanTransformerConfig) -> None:
        super().__init__()
        model_dim = int(config.model_dim)
        dropout = float(config.dropout)
        self.audio_from_dance = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=int(config.num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dance_from_audio = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=int(config.num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.audio_cross_norm = nn.LayerNorm(model_dim)
        self.dance_cross_norm = nn.LayerNorm(model_dim)
        self.audio_ffn_norm = nn.LayerNorm(model_dim)
        self.dance_ffn_norm = nn.LayerNorm(model_dim)
        self.audio_ffn = nn.Sequential(
            nn.Linear(model_dim, int(config.ffn_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config.ffn_dim), model_dim),
        )
        self.dance_ffn = nn.Sequential(
            nn.Linear(model_dim, int(config.ffn_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config.ffn_dim), model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_tokens: torch.Tensor, dance_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_cross, _ = self.audio_from_dance(audio_tokens, dance_tokens, dance_tokens, need_weights=False)
        dance_cross, _ = self.dance_from_audio(dance_tokens, audio_tokens, audio_tokens, need_weights=False)
        audio_tokens = self.audio_cross_norm(audio_tokens + self.dropout(audio_cross))
        dance_tokens = self.dance_cross_norm(dance_tokens + self.dropout(dance_cross))
        audio_tokens = self.audio_ffn_norm(audio_tokens + self.dropout(self.audio_ffn(audio_tokens)))
        dance_tokens = self.dance_ffn_norm(dance_tokens + self.dropout(self.dance_ffn(dance_tokens)))
        return audio_tokens, dance_tokens


class DiscreteShotPlanCMN(nn.Module):
    """Cross-modal network for frame-level shot-plan prediction.

    CMN keeps music and dance as two streams, repeatedly exchanges information
    through bidirectional cross-attention, then decodes shot slots from the
    fused memory plus previous/future keyframe timing context.
    """

    def __init__(self, config: ShotPlanTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.max_slots = int(config.max_slots)
        self.use_context_tokens = bool(config.use_context_tokens)
        self.future_context_size = int(config.future_context_size)
        model_dim = int(config.model_dim)

        self.audio_proj = nn.Linear(int(config.audio_dim), model_dim)
        self.dance_proj = nn.Linear(int(config.dance_dim), model_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, int(config.window_size), model_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 2, model_dim))
        self.input_dropout = nn.Dropout(float(config.input_dropout))
        self.blocks = nn.ModuleList(
            [_CrossModalFusionBlock(config) for _ in range(max(1, int(config.num_encoder_layers)))]
        )

        if self.use_context_tokens:
            self.context_type_embedding = nn.Parameter(torch.zeros(1, 3, model_dim))
            self.prev_position_embedding = nn.Parameter(torch.zeros(1, int(config.prev_context_size), 1, model_dim))
            self.prev_slot_embedding = nn.Parameter(torch.zeros(1, 1, int(config.max_slots), model_dim))
            self.cut_flag_proj = nn.Linear(1, model_dim)
            self.prev_delta_proj = nn.Linear(1, model_dim)
            if self.future_context_size > 0:
                self.future_position_embedding = nn.Parameter(torch.zeros(1, self.future_context_size, model_dim))
                self.future_delta_proj = nn.Linear(1, model_dim)
            self.prev_field_embeddings = nn.ModuleDict(
                {
                    field: nn.Embedding(int(vocab_size) + 1, model_dim, padding_idx=int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )

        self.memory_norm = nn.LayerNorm(model_dim)
        self.summary_norm = nn.LayerNorm(model_dim)
        self.summary_proj = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(model_dim, model_dim),
        )
        self.slot_queries = nn.Parameter(torch.zeros(1, int(config.max_slots), model_dim))
        self.slot_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(model_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(model_dim * 2, model_dim),
        )

        self.count_head = nn.Linear(model_dim, int(config.max_slots))
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

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        if self.use_context_tokens:
            nn.init.trunc_normal_(self.context_type_embedding, std=0.02)
            nn.init.trunc_normal_(self.prev_position_embedding, std=0.02)
            nn.init.trunc_normal_(self.prev_slot_embedding, std=0.02)
            if self.future_context_size > 0:
                nn.init.trunc_normal_(self.future_position_embedding, std=0.02)

    def _build_prev_tokens(
        self,
        prev_labels: dict[str, torch.Tensor] | None,
        prev_slot_mask: torch.Tensor | None,
        prev_frame_delta: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        prev_count = int(self.config.prev_context_size)
        if prev_labels is None or prev_slot_mask is None:
            return self.context_type_embedding[:, 1:2, :].expand(batch_size, prev_count, -1)

        prev_slot_mask = prev_slot_mask.to(device)
        label_repr = torch.zeros(
            batch_size,
            prev_count,
            self.max_slots,
            int(self.config.model_dim),
            device=device,
        )
        for field, embedding in self.prev_field_embeddings.items():
            labels = prev_labels[field].to(device)
            pad_id = int(self.config.vocab_sizes[field])
            labels = torch.where(labels < 0, torch.full_like(labels, pad_id), labels)
            label_repr = label_repr + embedding(labels)
        label_repr = label_repr / max(1, len(self.prev_field_embeddings))
        label_repr = label_repr + self.prev_position_embedding[:, :prev_count, :, :]
        label_repr = label_repr + self.prev_slot_embedding[:, :, : self.max_slots, :]

        mask = prev_slot_mask.unsqueeze(-1)
        prev_tokens = (label_repr * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        if prev_frame_delta is not None:
            delta = torch.log1p(prev_frame_delta.to(device).clamp_min(0.0)).unsqueeze(-1) / 5.0
            prev_tokens = prev_tokens + self.prev_delta_proj(delta)
        return prev_tokens + self.context_type_embedding[:, 1:2, :]

    def _build_future_tokens(
        self,
        future_frame_delta: torch.Tensor | None,
        future_frame_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        future_count = int(self.config.future_context_size)
        if future_count <= 0:
            return torch.zeros(batch_size, 0, int(self.config.model_dim), device=device)
        if future_frame_delta is None:
            delta = torch.zeros(batch_size, future_count, device=device)
        else:
            delta = torch.log1p(future_frame_delta.to(device).clamp_min(0.0)) / 5.0
        if future_frame_mask is None:
            mask = torch.zeros(batch_size, future_count, device=device)
        else:
            mask = future_frame_mask.to(device)
        future_tokens = self.future_delta_proj(delta.unsqueeze(-1))
        future_tokens = future_tokens + self.future_position_embedding[:, :future_count, :]
        future_tokens = future_tokens + self.context_type_embedding[:, 2:3, :]
        return future_tokens * mask.unsqueeze(-1)

    def _build_context_tokens(
        self,
        *,
        prev_labels: dict[str, torch.Tensor] | None,
        prev_slot_mask: torch.Tensor | None,
        prev_frame_delta: torch.Tensor | None,
        future_frame_delta: torch.Tensor | None,
        future_frame_mask: torch.Tensor | None,
        cut_flag: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_context_tokens:
            return torch.zeros(batch_size, 0, int(self.config.model_dim), device=device)
        if cut_flag is None:
            cut_flag = torch.zeros(batch_size, device=device)
        cut_token = self.cut_flag_proj(cut_flag.to(device).reshape(batch_size, 1, 1))
        cut_token = cut_token + self.context_type_embedding[:, 0:1, :]
        prev_tokens = self._build_prev_tokens(prev_labels, prev_slot_mask, prev_frame_delta, batch_size, device)
        future_tokens = self._build_future_tokens(future_frame_delta, future_frame_mask, batch_size, device)
        return torch.cat([cut_token, prev_tokens, future_tokens], dim=1)

    def _build_outputs(
        self,
        summary: torch.Tensor,
        slot_repr: torch.Tensor,
        cut_flag: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        del cut_flag
        outputs = {"count_logits": self.count_head(summary)}
        for field, head in self.field_heads.items():
            outputs[f"{field}_logits"] = head(slot_repr)
        for name, head in self.joint_heads.items():
            outputs[f"{name}_logits"] = head(slot_repr)
        return outputs

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        prev_labels: dict[str, torch.Tensor] | None = None,
        prev_slot_mask: torch.Tensor | None = None,
        prev_frame_delta: torch.Tensor | None = None,
        future_frame_delta: torch.Tensor | None = None,
        future_frame_mask: torch.Tensor | None = None,
        cut_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        summary, slot_repr = self._encode_representations(
            audio,
            dance,
            prev_labels=prev_labels,
            prev_slot_mask=prev_slot_mask,
            prev_frame_delta=prev_frame_delta,
            future_frame_delta=future_frame_delta,
            future_frame_mask=future_frame_mask,
            cut_flag=cut_flag,
        )
        return self._build_outputs(summary, slot_repr, cut_flag)

    def _encode_representations(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        prev_labels: dict[str, torch.Tensor] | None = None,
        prev_slot_mask: torch.Tensor | None = None,
        prev_frame_delta: torch.Tensor | None = None,
        future_frame_delta: torch.Tensor | None = None,
        future_frame_mask: torch.Tensor | None = None,
        cut_flag: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, window_size, _ = audio.shape
        device = audio.device

        time_embed = self.time_embedding[:, :window_size, :]
        audio_tokens = self.audio_proj(audio) + time_embed + self.modality_embedding[:, 0:1, :]
        dance_tokens = self.dance_proj(dance) + time_embed + self.modality_embedding[:, 1:2, :]
        audio_tokens = self.input_dropout(audio_tokens)
        dance_tokens = self.input_dropout(dance_tokens)

        for block in self.blocks:
            audio_tokens, dance_tokens = block(audio_tokens, dance_tokens)

        context_tokens = self._build_context_tokens(
            prev_labels=prev_labels,
            prev_slot_mask=prev_slot_mask,
            prev_frame_delta=prev_frame_delta,
            future_frame_delta=future_frame_delta,
            future_frame_mask=future_frame_mask,
            cut_flag=cut_flag,
            batch_size=batch_size,
            device=device,
        )
        memory = self.memory_norm(torch.cat([audio_tokens, dance_tokens, context_tokens], dim=1))
        audio_pool = audio_tokens.mean(dim=1)
        dance_pool = dance_tokens.mean(dim=1)
        memory_pool = memory.mean(dim=1)
        summary = self.summary_norm(memory_pool + self.summary_proj(torch.cat([audio_pool, dance_pool, memory_pool], dim=-1)))

        slot_queries = self.slot_queries.expand(batch_size, -1, -1)
        slot_repr, _ = self.slot_attention(slot_queries, memory, memory, need_weights=False)
        slot_repr = self.slot_norm(slot_repr + summary.unsqueeze(1))
        slot_repr = slot_repr + self.slot_ffn(slot_repr)

        return summary, slot_repr


class DiscreteShotPlanCMNHeadMoE(DiscreteShotPlanCMN):
    """CMN with cut-conditioned output heads.

    The cross-modal backbone is shared, while the final count and field heads
    are split into non-cut and cut experts selected by the binary cut flag.
    """

    def __init__(self, config: ShotPlanTransformerConfig) -> None:
        super().__init__(config)
        model_dim = int(config.model_dim)
        del self.count_head
        del self.field_heads

        self.noncut_count_head = nn.Linear(model_dim, int(config.max_slots))
        self.cut_count_head = nn.Linear(model_dim, int(config.max_slots))
        self.noncut_field_heads = nn.ModuleDict(
            {
                field: nn.Linear(model_dim, int(vocab_size))
                for field, vocab_size in config.vocab_sizes.items()
            }
        )
        self.cut_field_heads = nn.ModuleDict(
            {
                field: nn.Linear(model_dim, int(vocab_size))
                for field, vocab_size in config.vocab_sizes.items()
            }
        )

    @staticmethod
    def _mix_by_cut_flag(
        noncut_logits: torch.Tensor,
        cut_logits: torch.Tensor,
        cut_flag: torch.Tensor | None,
    ) -> torch.Tensor:
        if cut_flag is None:
            return noncut_logits
        gate_shape = [int(cut_flag.shape[0]), *([1] * (noncut_logits.ndim - 1))]
        gate = cut_flag.to(noncut_logits.device).reshape(gate_shape).clamp(0.0, 1.0)
        return noncut_logits * (1.0 - gate) + cut_logits * gate

    def _build_outputs(
        self,
        summary: torch.Tensor,
        slot_repr: torch.Tensor,
        cut_flag: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        outputs = {
            "count_logits": self._mix_by_cut_flag(
                self.noncut_count_head(summary),
                self.cut_count_head(summary),
                cut_flag,
            )
        }
        for field in self.noncut_field_heads:
            outputs[f"{field}_logits"] = self._mix_by_cut_flag(
                self.noncut_field_heads[field](slot_repr),
                self.cut_field_heads[field](slot_repr),
                cut_flag,
            )
        for name, head in getattr(self, "joint_heads", {}).items():
            outputs[f"{name}_logits"] = head(slot_repr)
        return outputs


class DiscreteShotPlanMLP(nn.Module):
    """Simple non-attention baseline for discrete shot-plan prediction.

    The MLP intentionally avoids both Transformer attention and FiLM fusion. It
    flattens the 5fps music/dance window, appends cut/previous-keyframe context,
    and predicts the slot labels directly from a feed-forward network.
    """

    def __init__(self, config: ShotPlanTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.max_slots = int(config.max_slots)
        self.use_context_tokens = bool(config.use_context_tokens)
        self.future_context_size = int(config.future_context_size)
        hidden_dim = int(config.model_dim)

        input_dim = int(config.window_size) * (int(config.audio_dim) + int(config.dance_dim))
        if self.use_context_tokens:
            self.prev_field_embeddings = nn.ModuleDict(
                {
                    field: nn.Embedding(int(vocab_size) + 1, hidden_dim, padding_idx=int(vocab_size))
                    for field, vocab_size in config.vocab_sizes.items()
                }
            )
            input_dim += 1  # cut_flag
            input_dim += int(config.prev_context_size) * hidden_dim
            input_dim += int(config.prev_context_size)  # log-scaled prev frame deltas
            input_dim += int(config.prev_context_size) * int(config.max_slots)  # prev slot mask
            input_dim += self.future_context_size  # log-scaled future frame deltas
            input_dim += self.future_context_size  # future timing mask

        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Dropout(float(config.input_dropout)),
        ]
        layer_count = max(1, int(config.num_encoder_layers))
        current_dim = input_dim
        for _ in range(layer_count):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.count_head = nn.Linear(hidden_dim, int(config.max_slots))
        self.slot_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(int(config.max_slots))
            ]
        )
        self.field_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, int(vocab_size))
                for field, vocab_size in config.vocab_sizes.items()
            }
        )
        self.joint_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, int(vocab_size))
                for name, vocab_size in (config.joint_vocab_sizes or {}).items()
            }
        )

    def _build_prev_context(
        self,
        prev_labels: dict[str, torch.Tensor] | None,
        prev_slot_mask: torch.Tensor | None,
        prev_frame_delta: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        prev_count = int(self.config.prev_context_size)
        hidden_dim = int(self.config.model_dim)
        max_slots = int(self.config.max_slots)
        if prev_labels is None or prev_slot_mask is None:
            prev_repr = torch.zeros(batch_size, prev_count, hidden_dim, device=device)
            prev_mask = torch.zeros(batch_size, prev_count, max_slots, device=device)
        else:
            prev_mask = prev_slot_mask.to(device)
            label_repr = torch.zeros(batch_size, prev_count, max_slots, hidden_dim, device=device)
            for field, embedding in self.prev_field_embeddings.items():
                labels = prev_labels[field].to(device)
                pad_id = int(self.config.vocab_sizes[field])
                labels = torch.where(labels < 0, torch.full_like(labels, pad_id), labels)
                label_repr = label_repr + embedding(labels)
            label_repr = label_repr / max(1, len(self.prev_field_embeddings))
            mask = prev_mask.unsqueeze(-1)
            prev_repr = (label_repr * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

        if prev_frame_delta is None:
            delta = torch.zeros(batch_size, prev_count, device=device)
        else:
            delta = torch.log1p(prev_frame_delta.to(device).clamp_min(0.0)) / 5.0
        return torch.cat(
            [
                prev_repr.reshape(batch_size, -1),
                delta.reshape(batch_size, -1),
                prev_mask.reshape(batch_size, -1),
            ],
            dim=-1,
        )

    def _build_future_context(
        self,
        future_frame_delta: torch.Tensor | None,
        future_frame_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        future_count = int(self.config.future_context_size)
        if future_count <= 0:
            return torch.zeros(batch_size, 0, device=device)
        if future_frame_delta is None:
            delta = torch.zeros(batch_size, future_count, device=device)
        else:
            delta = torch.log1p(future_frame_delta.to(device).clamp_min(0.0)) / 5.0
        if future_frame_mask is None:
            mask = torch.zeros(batch_size, future_count, device=device)
        else:
            mask = future_frame_mask.to(device)
        return torch.cat([delta * mask, mask], dim=-1)

    def forward(
        self,
        audio: torch.Tensor,
        dance: torch.Tensor,
        *,
        prev_labels: dict[str, torch.Tensor] | None = None,
        prev_slot_mask: torch.Tensor | None = None,
        prev_frame_delta: torch.Tensor | None = None,
        future_frame_delta: torch.Tensor | None = None,
        future_frame_mask: torch.Tensor | None = None,
        cut_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(audio.shape[0])
        device = audio.device
        pieces = [
            audio.reshape(batch_size, -1),
            dance.reshape(batch_size, -1),
        ]
        if self.use_context_tokens:
            if cut_flag is None:
                cut_flag = torch.zeros(batch_size, device=device)
            pieces.append(cut_flag.to(device).reshape(batch_size, 1))
            pieces.append(
                self._build_prev_context(
                    prev_labels,
                    prev_slot_mask,
                    prev_frame_delta,
                    batch_size,
                    device,
                )
            )
            pieces.append(self._build_future_context(future_frame_delta, future_frame_mask, batch_size, device))

        hidden = self.backbone(torch.cat(pieces, dim=-1))
        slot_repr = torch.stack([head(hidden) for head in self.slot_heads], dim=1)

        outputs = {"count_logits": self.count_head(hidden)}
        for field, head in self.field_heads.items():
            outputs[f"{field}_logits"] = head(slot_repr)
        for name, head in self.joint_heads.items():
            outputs[f"{name}_logits"] = head(slot_repr)
        return outputs
