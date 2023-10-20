import math
import torch
from torch import nn
from typing import Optional, Tuple
from dataclasses import dataclass
from argparse import ArgumentParser
import pandas as pd
from Bio import SeqIO
from typing import Union, List


@dataclass
class Config:
    vocab_size: int = 25
    mask_token_id: int = 32
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-05
    hidden_size: int = 1280
    pad_token_id: int = 1
    max_position_embeddings: int = 1024
    chunk_size_feed_forward: int = 0


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        torch.float32
    ).min
    return extended_attention_mask


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id
        self.mask_token_id = config.mask_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        embeddings = embeddings * (1 - 0.15 * 0.8) / (1 - 0.0)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(
                embeddings.dtype
            )
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size**-0.5
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        return hidden_states


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask
        )
        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, 33, bias=False)
        self.bias = nn.Parameter(torch.zeros(33))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class ForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)
        self.lm_head = LMHead(config)
        self.VOCAB = {c: i + 4 for i, c in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO")}
        self.VOCAB = {"sos": 0, "eos": 2, "pad": 1, "unk": 3, **self.VOCAB}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        sequence_output = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(sequence_output)
        return logits

    def tokenize(self, sequence: Union[str, List[str]]):
        if isinstance(sequence, str):
            sequence_ids = (
                [
                    self.VOCAB["sos"],
                ]
                + [self.VOCAB[c] for c in sequence]
                + [
                    self.VOCAB["eos"],
                ]
            )
            sequence_ids = torch.tensor([sequence_ids], dtype=torch.long)
        else:
            sequence_ids = []
            for seq in sequence:
                ids = (
                    [
                        self.VOCAB["sos"],
                    ]
                    + [self.VOCAB[c] for c in seq]
                    + [
                        self.VOCAB["eos"],
                    ]
                )
                sequence_ids.append(torch.tensor(ids, dtype=torch.long))
            # PAD
            max_len = max([len(seq) for seq in sequence_ids])
            for i in range(len(sequence_ids)):
                sequence_ids[i] = torch.cat(
                    [
                        sequence_ids[i],
                        torch.ones(max_len - len(sequence_ids[i]), dtype=torch.long)
                        * self.VOCAB["pad"],
                    ]
                )
            sequence_ids = torch.stack(sequence_ids)
        return sequence_ids
