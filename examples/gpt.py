from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, SupportsInt, TypeAlias, cast

import monumpy as np

RNG: TypeAlias = np.random.Generator


class Array(Protocol):
  @property
  def T(self) -> Array: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  def __getitem__(self, key: object) -> object: ...
  def __add__(self, other: object) -> Array: ...
  def __radd__(self, other: object) -> Array: ...
  def __sub__(self, other: object) -> Array: ...
  def __rsub__(self, other: object) -> Array: ...
  def __mul__(self, other: object) -> Array: ...
  def __rmul__(self, other: object) -> Array: ...
  def __truediv__(self, other: object) -> Array: ...
  def __rtruediv__(self, other: object) -> Array: ...
  def __matmul__(self, other: object) -> Array: ...
  def __rmatmul__(self, other: object) -> Array: ...


def _a(x: object) -> Array: return cast(Array, x)


@dataclass(frozen=True, slots=True)
class GPTConfig:
  vocab_size: int = 64
  context_size: int = 8
  hidden_size: int = 16
  mlp_size: int = 64


@dataclass(frozen=True, slots=True)
class GPTParams:
  token_embedding: Array
  position_embedding: Array
  w_q: Array
  w_k: Array
  w_v: Array
  w_o: Array
  ln1_gain: Array
  ln1_bias: Array
  mlp_in: Array
  mlp_out: Array
  ln2_gain: Array
  ln2_bias: Array
  final_ln_gain: Array
  final_ln_bias: Array
  lm_head: Array
  causal_mask: Array


def _normal(rng: RNG, shape: tuple[int, ...], *, scale: float = 0.02) -> Array:
  return _a(np.asarray(rng.normal(0.0, scale, size=shape), dtype=np.float32))


def _ones(shape: tuple[int, ...]) -> Array:
  return _a(np.ones(shape, dtype=np.float32))


def _zeros(shape: tuple[int, ...]) -> Array:
  return _a(np.zeros(shape, dtype=np.float32))


def init_params(config: GPTConfig, *, seed: int = 0) -> GPTParams:
  rng = np.random.default_rng(seed)
  hidden, ctx = config.hidden_size, config.context_size
  return GPTParams(
    token_embedding=_normal(rng, (config.vocab_size, hidden)),
    position_embedding=_normal(rng, (ctx, hidden)),
    w_q=_normal(rng, (hidden, hidden)),
    w_k=_normal(rng, (hidden, hidden)),
    w_v=_normal(rng, (hidden, hidden)),
    w_o=_normal(rng, (hidden, hidden)),
    ln1_gain=_ones((hidden,)),
    ln1_bias=_zeros((hidden,)),
    mlp_in=_normal(rng, (hidden, config.mlp_size)),
    mlp_out=_normal(rng, (config.mlp_size, hidden)),
    ln2_gain=_ones((hidden,)),
    ln2_bias=_zeros((hidden,)),
    final_ln_gain=_ones((hidden,)),
    final_ln_bias=_zeros((hidden,)),
    lm_head=_normal(rng, (hidden, config.vocab_size)),
    causal_mask=_a(np.triu(np.ones((ctx, ctx), dtype=np.bool), k=1)),
  )


def embedding_lookup(table: Array, ids: tuple[int, ...]) -> Array:
  return _a(np.stack([_a(table[token]) for token in ids]))


def softmax(x: Array) -> Array:
  shifted = x - _a(np.max(x, axis=-1, keepdims=True))
  weights = _a(np.exp(shifted))
  return weights / _a(np.sum(weights, axis=-1, keepdims=True))


def gelu(x: Array) -> Array:
  cubic = _a(np.power(x, 3.0))
  inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * cubic)
  return 0.5 * x * (1.0 + _a(np.tanh(inner)))


def layer_norm(x: Array, gain: Array, bias: Array, *, eps: float = 1e-5) -> Array:
  mean = _a(np.mean(x, axis=-1, keepdims=True))
  centered = x - mean
  variance = _a(np.mean(centered * centered, axis=-1, keepdims=True))
  return centered / _a(np.sqrt(variance + eps)) * gain + bias


def self_attn(
  x: Array,
  w_q: Array,
  w_k: Array,
  w_v: Array,
  w_o: Array,
  causal_mask: Array,
) -> Array:
  q = x @ w_q
  k = x @ w_k
  v = x @ w_v
  scores = (q @ k.T) / math.sqrt(q.shape[-1])
  mask = _a(causal_mask[: x.shape[0], : x.shape[0]])
  fill = _a(np.full(scores.shape, -1.0e9, dtype=np.float32))
  scores = _a(np.where(mask, fill, scores))
  return (softmax(scores) @ v) @ w_o


def forward(tokens: tuple[int, ...], params: GPTParams) -> Array:
  positions = tuple(range(len(tokens)))
  x = embedding_lookup(params.token_embedding, tokens) + embedding_lookup(
    params.position_embedding,
    positions,
  )

  attn_input = layer_norm(x, params.ln1_gain, params.ln1_bias)
  x = x + self_attn(
    attn_input,
    params.w_q,
    params.w_k,
    params.w_v,
    params.w_o,
    params.causal_mask,
  )

  mlp_input = layer_norm(x, params.ln2_gain, params.ln2_bias)
  x = x + gelu(mlp_input @ params.mlp_in) @ params.mlp_out

  x = layer_norm(x, params.final_ln_gain, params.final_ln_bias)
  return x @ params.lm_head


def main() -> None:
  config = GPTConfig()
  params = init_params(config, seed=420)
  logits = forward((3, 1, 4, 1, 5, 9, 2, 6), params)
  next_token = int(cast(SupportsInt, np.argmax(_a(logits[-1]))))
  print(f"logits shape: {logits.shape}")
  print(f"next-token argmax: {next_token}")


if __name__ == "__main__":
  main()
