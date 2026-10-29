"""MiniMind-3 decoder-only model and MoE variant.

The implementation follows the reference MiniMind architecture while exposing a
standard Hugging Face causal-LM interface so it can share the rest of ``lmml``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    """Configuration for the dense and top-k MoE MiniMind models."""

    model_type = "minimind"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.use_moe = bool(use_moe)
        self.dropout = float(kwargs.get("dropout", 0.0))
        self.vocab_size = int(kwargs.get("vocab_size", 6400))
        self.bos_token_id = int(kwargs.get("bos_token_id", 1))
        self.eos_token_id = int(kwargs.get("eos_token_id", 2))
        self.flash_attn = bool(kwargs.get("flash_attn", True))
        self.num_attention_heads = int(kwargs.get("num_attention_heads", 8))
        self.num_key_value_heads = int(kwargs.get("num_key_value_heads", 4))
        self.head_dim = int(kwargs.get("head_dim", self.hidden_size // self.num_attention_heads))
        self.hidden_act = str(kwargs.get("hidden_act", "silu"))
        default_intermediate = math.ceil(self.hidden_size * math.pi / 64) * 64
        self.intermediate_size = int(kwargs.get("intermediate_size", default_intermediate))
        self.max_position_embeddings = int(kwargs.get("max_position_embeddings", 32768))
        self.rms_norm_eps = float(kwargs.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(kwargs.get("rope_theta", 1e6))
        self.tie_word_embeddings = bool(kwargs.get("tie_word_embeddings", True))
        inference_rope_scaling = bool(kwargs.get("inference_rope_scaling", False))
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )
        self.num_experts = int(kwargs.get("num_experts", 4))
        self.num_experts_per_tok = int(kwargs.get("num_experts_per_tok", 1))
        self.moe_intermediate_size = int(
            kwargs.get("moe_intermediate_size", self.intermediate_size)
        )
        self.norm_topk_prob = bool(kwargs.get("norm_topk_prob", True))
        self.router_aux_loss_coef = float(kwargs.get("router_aux_loss_coef", 5e-4))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(self.weight.dtype)


def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cosine/sine tables, optionally with YaRN scaling."""
    freqs = 1.0 / (
        float(rope_base) ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    )
    attention_factor = 1.0
    if rope_scaling is not None:
        original_max = float(rope_scaling.get("original_max_position_embeddings", 2048))
        factor = float(rope_scaling.get("factor", 16))
        beta_fast = float(rope_scaling.get("beta_fast", 32))
        beta_slow = float(rope_scaling.get("beta_slow", 1))
        attention_factor = float(rope_scaling.get("attention_factor", 1.0))
        if end / original_max > 1.0:
            inv_dim = lambda beta: (dim * math.log(original_max / (beta * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    positions = torch.arange(end, device=freqs.device, dtype=torch.float32)
    angles = torch.outer(positions, freqs).float()
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1) * attention_factor
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1) * attention_factor
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq_len, kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, kv_heads * n_rep, head_dim)
    )


class MiniMindAttention(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.use_sdpa = bool(config.flash_attn)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[1]
            k = torch.cat((past_key_value[0], k), dim=1)
            v = torch.cat((past_key_value[1], v), dim=1)
        present = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
        v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
        key_len = k.shape[-2]

        # SDPA's is_causal fast path is valid only when no cache/padding is present.
        if (
            self.use_sdpa
            and seq_len > 1
            and past_len == 0
            and (attention_mask is None or bool(torch.all(attention_mask == 1)))
        ):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            query_positions = torch.arange(
                past_len, past_len + seq_len, device=scores.device
            ).unsqueeze(1)
            key_positions = torch.arange(key_len, device=scores.device).unsqueeze(0)
            causal = key_positions <= query_positions
            scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
            if attention_mask is not None:
                # Accept [B, K] masks. When a caller supplies an already sliced
                # mask during generation, align it to the current query length.
                key_mask = attention_mask[:, -key_len:].bool()
                scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
            output = self.attn_dropout(F.softmax(scores, dim=-1).to(q.dtype)) @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), present


class MiniMindFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniMindMoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MiniMindFeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        flat = x.reshape(-1, hidden_size)
        scores = F.softmax(self.gate(flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
            else:
                # Straight-through scalar gate: forward weight 1, aux loss trains routing.
                top1 = torch.topk(
                    F.softmax(self.gate(flat.detach()), dim=-1), k=1, dim=-1, sorted=False
                )[0]
                topk_weight = top1 - top1.detach() + 1.0

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            mask = topk_idx == expert_index
            if bool(mask.any()):
                token_indices = mask.any(dim=-1).nonzero(as_tuple=False).flatten()
                weights = topk_weight[mask].view(-1, 1)
                output.index_add_(
                    0,
                    token_indices,
                    (expert(flat[token_indices]) * weights).to(output.dtype),
                )
            elif self.training:
                # Keep every expert in the autograd graph for DDP and lazy backends.
                output[0, 0] += 0 * sum(parameter.sum() for parameter in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = scores.new_zeros(())
        return output.view(batch_size, seq_len, hidden_size)


class MiniMindBlock(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.self_attn = MiniMindAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            MiniMindFeedForward(config) if not config.use_moe else MiniMindMoEFeedForward(config)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = precompute_freqs_cis(
            config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None, torch.Tensor]:
        _, seq_len = input_ids.shape
        past = list(past_key_values) if past_key_values is not None else [None] * len(self.layers)
        start_pos = past[0][0].shape[1] if past and past[0] is not None else 0
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if (
            not bool(torch.isfinite(freqs_cos).all())
            or abs(float(freqs_cos[0, 0].item()) - 1.0) > 0.5
        ):
            freqs_cos, freqs_sin = precompute_freqs_cis(
                self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            freqs_cos, freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
            # Never retain tensors created under ``inference_mode`` as module
            # buffers: PPO later reuses the same model in an autograd forward.
            if not torch.is_inference_mode_enabled():
                self.freqs_cos, self.freqs_sin = freqs_cos, freqs_sin
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if start_pos + seq_len > self.freqs_cos.shape[0]:
            raise ValueError(
                f"sequence position {start_pos + seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        positions = (freqs_cos[start_pos : start_pos + seq_len], freqs_sin[start_pos : start_pos + seq_len])
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        use_checkpoint = bool(
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and past[0] is None
        )
        for layer, layer_past in zip(self.layers, past):
            if use_checkpoint:
                def _checkpointed(
                    current: torch.Tensor,
                    block: MiniMindBlock = layer,
                    layer_mask: torch.Tensor | None = attention_mask,
                ) -> tuple[torch.Tensor, Any]:
                    return block(
                        current,
                        positions,
                        past_key_value=None,
                        use_cache=False,
                        attention_mask=layer_mask,
                    )

                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    _checkpointed,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    positions,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
            if present is not None:
                presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_losses = [
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MiniMindMoEFeedForward)
        ]
        aux_loss = torch.stack(aux_losses).sum() if aux_losses else hidden_states.new_zeros(())
        return hidden_states, (presents if use_cache else None), aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    base_model_prefix = "model"
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.embed_tokens.weight"
    }
    _supports_cache_class = False
    _supports_sdpa = True
    _supports_gradient_checkpointing = True
    supports_gradient_checkpointing = True

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MoeCausalLMOutputWithPast:
        hidden_states, presents, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Minimal deterministic/sampling decoder with KV cache.

        ``GenerativeLM``'s transformers cache protocol changes across releases;
        MiniMind keeps its own compact KV-cache loop for stable behavior.
        """
        if "input_ids" in kwargs:
            inputs = kwargs["input_ids"]
        if inputs is None:
            raise ValueError("generate requires inputs or input_ids")
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        eos = self.config.eos_token_id if eos_token_id is None else int(eos_token_id)
        config_pad = getattr(self.config, "pad_token_id", None)
        pad = config_pad if pad_token_id is None else int(pad_token_id)

        generated = inputs
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        finished = torch.zeros(inputs.size(0), dtype=torch.bool, device=inputs.device)
        for _ in range(max(0, int(max_new_tokens))):
            model_input = generated if past is None else generated[:, -1:]
            generated.shape[1]
            outputs = self(
                model_input,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            if do_sample and temperature > 0:
                logits = logits / float(temperature)
            if repetition_penalty and repetition_penalty != 1.0:
                for row in range(generated.size(0)):
                    seen = torch.unique(generated[row])
                    score = logits[row, seen]
                    logits[row, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )
            if top_k and top_k > 0 and top_k < logits.size(-1):
                threshold = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs > float(top_p)
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_indices, sorted_logits
                )
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if pad is not None:
                next_token = torch.where(
                    finished[:, None],
                    next_token.new_full((next_token.size(0), 1), pad),
                    next_token,
                )
            generated = torch.cat((generated, next_token), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))), dim=1
                )
            finished |= next_token.squeeze(-1).eq(eos)
            if bool(finished.all()):
                break
        return generated


class MiniMindCriticForCausalLM(MiniMindForCausalLM):
    """MiniMind policy/critic shared trunk with a token-level scalar value head."""

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states, _, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.value_head(hidden_states).squeeze(-1)


__all__ = [
    "MiniMindConfig",
    "MiniMindCriticForCausalLM",
    "MiniMindForCausalLM",
    "MiniMindModel",
    "precompute_freqs_cis",
]
