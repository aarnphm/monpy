from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import monumpy as mnp
import numpy as np
import numpy.typing as npt

from monpy._bench.core import BenchCase
from monpy._bench.types._helpers import prefix_groups

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class MonpyArray(Protocol):
  @property
  def T(self) -> MonpyArray: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  def __add__(self, other: object) -> MonpyArray: ...
  def __radd__(self, other: object) -> MonpyArray: ...
  def __sub__(self, other: object) -> MonpyArray: ...
  def __rsub__(self, other: object) -> MonpyArray: ...
  def __mul__(self, other: object) -> MonpyArray: ...
  def __rmul__(self, other: object) -> MonpyArray: ...
  def __truediv__(self, other: object) -> MonpyArray: ...
  def __rtruediv__(self, other: object) -> MonpyArray: ...
  def __matmul__(self, other: object) -> MonpyArray: ...
  def __rmatmul__(self, other: object) -> MonpyArray: ...


def _m(x: object) -> MonpyArray:
  return cast(MonpyArray, x)


@dataclass(frozen=True, slots=True)
class NumpyAttentionParams:
  w_q: FloatArray
  w_k: FloatArray
  w_v: FloatArray
  w_o: FloatArray
  ln1_gain: FloatArray
  ln1_bias: FloatArray
  mlp_in: FloatArray
  mlp_out: FloatArray
  ln2_gain: FloatArray
  ln2_bias: FloatArray
  final_ln_gain: FloatArray
  final_ln_bias: FloatArray
  lm_head: FloatArray
  causal_mask: BoolArray


@dataclass(frozen=True, slots=True)
class MonpyAttentionParams:
  w_q: MonpyArray
  w_k: MonpyArray
  w_v: MonpyArray
  w_o: MonpyArray
  ln1_gain: MonpyArray
  ln1_bias: MonpyArray
  mlp_in: MonpyArray
  mlp_out: MonpyArray
  ln2_gain: MonpyArray
  ln2_bias: MonpyArray
  final_ln_gain: MonpyArray
  final_ln_bias: MonpyArray
  lm_head: MonpyArray
  causal_mask: MonpyArray


def _normal(rng: np.random.Generator, shape: tuple[int, ...], *, scale: float = 0.02) -> FloatArray:
  return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _softmax_numpy(scores: FloatArray) -> FloatArray:
  shifted = scores - np.max(scores, axis=-1, keepdims=True)
  weights = np.exp(shifted)
  return weights / np.sum(weights, axis=-1, keepdims=True)


def _softmax_monpy(scores: MonpyArray) -> MonpyArray:
  return _m(mnp.nn.softmax(scores, axis=-1))


def _gelu_numpy(x: FloatArray) -> FloatArray:
  inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3.0))
  return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_monpy(x: MonpyArray) -> MonpyArray:
  cubic = _m(mnp.power(x, 3.0))
  inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * cubic)
  return 0.5 * x * (1.0 + _m(mnp.tanh(inner)))


def _layer_norm_numpy(
  x: FloatArray,
  gain: FloatArray,
  bias: FloatArray,
  *,
  eps: float = 1e-5,
) -> FloatArray:
  mean = np.mean(x, axis=-1, keepdims=True)
  centered = x - mean
  variance = np.mean(centered * centered, axis=-1, keepdims=True)
  return centered / np.sqrt(variance + eps) * gain + bias


def _layer_norm_monpy(
  x: MonpyArray,
  gain: MonpyArray,
  bias: MonpyArray,
  *,
  eps: float = 1e-5,
) -> MonpyArray:
  return _m(mnp.nn.layer_norm(x, gain, bias, eps))


def _causal_attention_numpy(
  x: FloatArray,
  w_q: FloatArray,
  w_k: FloatArray,
  w_v: FloatArray,
  w_o: FloatArray,
  causal_mask: BoolArray,
) -> FloatArray:
  q = x @ w_q
  k = x @ w_k
  v = x @ w_v
  scores = (q @ k.T) / math.sqrt(q.shape[-1])
  scores = np.where(causal_mask, np.full_like(scores, -1.0e9), scores)
  return (_softmax_numpy(scores) @ v) @ w_o


def _causal_attention_monpy(
  x: MonpyArray,
  w_q: MonpyArray,
  w_k: MonpyArray,
  w_v: MonpyArray,
  w_o: MonpyArray,
  causal_mask: MonpyArray,
) -> MonpyArray:
  q = x @ w_q
  k = x @ w_k
  v = x @ w_v
  scores = q @ k.T
  weights = _m(mnp.nn.scaled_masked_softmax(scores, causal_mask, 1.0 / math.sqrt(q.shape[-1])))
  return (weights @ v) @ w_o


def _gpt_logits_numpy(x: FloatArray, params: NumpyAttentionParams) -> FloatArray:
  attn_input = _layer_norm_numpy(x, params.ln1_gain, params.ln1_bias)
  x = x + _causal_attention_numpy(
    attn_input,
    params.w_q,
    params.w_k,
    params.w_v,
    params.w_o,
    params.causal_mask,
  )
  mlp_input = _layer_norm_numpy(x, params.ln2_gain, params.ln2_bias)
  x = x + _gelu_numpy(mlp_input @ params.mlp_in) @ params.mlp_out
  x = _layer_norm_numpy(x, params.final_ln_gain, params.final_ln_bias)
  return x @ params.lm_head


def _gpt_logits_monpy(x: MonpyArray, params: MonpyAttentionParams) -> MonpyArray:
  attn_input = _layer_norm_monpy(x, params.ln1_gain, params.ln1_bias)
  x = x + _causal_attention_monpy(
    attn_input,
    params.w_q,
    params.w_k,
    params.w_v,
    params.w_o,
    params.causal_mask,
  )
  mlp_input = _layer_norm_monpy(x, params.ln2_gain, params.ln2_bias)
  x = x + _gelu_monpy(mlp_input @ params.mlp_in) @ params.mlp_out
  x = _layer_norm_monpy(x, params.final_ln_gain, params.final_ln_bias)
  return x @ params.lm_head


def _monpy_array(value: FloatArray | BoolArray) -> MonpyArray:
  dtype = mnp.bool if value.dtype == np.bool_ else mnp.float32
  return _m(mnp.asarray(value, dtype=dtype, copy=False))


def _monpy_params(params: NumpyAttentionParams) -> MonpyAttentionParams:
  return MonpyAttentionParams(
    w_q=_monpy_array(params.w_q),
    w_k=_monpy_array(params.w_k),
    w_v=_monpy_array(params.w_v),
    w_o=_monpy_array(params.w_o),
    ln1_gain=_monpy_array(params.ln1_gain),
    ln1_bias=_monpy_array(params.ln1_bias),
    mlp_in=_monpy_array(params.mlp_in),
    mlp_out=_monpy_array(params.mlp_out),
    ln2_gain=_monpy_array(params.ln2_gain),
    ln2_bias=_monpy_array(params.ln2_bias),
    final_ln_gain=_monpy_array(params.final_ln_gain),
    final_ln_bias=_monpy_array(params.final_ln_bias),
    lm_head=_monpy_array(params.lm_head),
    causal_mask=_monpy_array(params.causal_mask),
  )


def build_cases(
  *,
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  del vector_sizes, linalg_sizes
  seq = max(8, min(max(matrix_sizes), 32))
  hidden = max(8, min(vector_size // 32, 32))
  mlp = hidden * 4
  vocab = max(32, min(vector_size // 8, 128))

  rng = np.random.default_rng(20260508)
  x_np = _normal(rng, (seq, hidden))
  params_np = NumpyAttentionParams(
    w_q=_normal(rng, (hidden, hidden)),
    w_k=_normal(rng, (hidden, hidden)),
    w_v=_normal(rng, (hidden, hidden)),
    w_o=_normal(rng, (hidden, hidden)),
    ln1_gain=np.ones((hidden,), dtype=np.float32),
    ln1_bias=np.zeros((hidden,), dtype=np.float32),
    mlp_in=_normal(rng, (hidden, mlp)),
    mlp_out=_normal(rng, (mlp, hidden)),
    ln2_gain=np.ones((hidden,), dtype=np.float32),
    ln2_bias=np.zeros((hidden,), dtype=np.float32),
    final_ln_gain=np.ones((hidden,), dtype=np.float32),
    final_ln_bias=np.zeros((hidden,), dtype=np.float32),
    lm_head=_normal(rng, (hidden, vocab)),
    causal_mask=np.triu(np.ones((seq, seq), dtype=np.bool_), k=1),
  )
  x_mp = _m(mnp.asarray(x_np, dtype=mnp.float32, copy=False))
  params_mp = _monpy_params(params_np)

  scores_np = (x_np @ x_np.T) / math.sqrt(hidden)
  scores_np = np.where(
    params_np.causal_mask,
    np.full_like(scores_np, -1.0e9),
    scores_np,
  )
  scores_mp = (x_mp @ x_mp.T) / math.sqrt(hidden)
  scores_mp = _m(
    mnp.where(params_mp.causal_mask, _m(mnp.full(scores_mp.shape, -1.0e9, dtype=mnp.float32)), scores_mp),
  )

  cases = [
    BenchCase(
      "softmax",
      f"causal_scores_t{seq}_f32",
      lambda scores=scores_mp: _softmax_monpy(scores),
      lambda scores=scores_np: _softmax_numpy(scores),
    ),
    BenchCase(
      "attention",
      f"causal_attention_t{seq}_d{hidden}_f32",
      lambda x=x_mp, p=params_mp: _causal_attention_monpy(
        x,
        p.w_q,
        p.w_k,
        p.w_v,
        p.w_o,
        p.causal_mask,
      ),
      lambda x=x_np, p=params_np: _causal_attention_numpy(
        x,
        p.w_q,
        p.w_k,
        p.w_v,
        p.w_o,
        p.causal_mask,
      ),
      rtol=1e-4,
      atol=1e-4,
    ),
    BenchCase(
      "gpt",
      f"tiny_gpt_logits_t{seq}_d{hidden}_v{vocab}_f32",
      lambda x=x_mp, p=params_mp: _gpt_logits_monpy(x, p),
      lambda x=x_np, p=params_np: _gpt_logits_numpy(x, p),
      rtol=1e-4,
      atol=1e-4,
    ),
  ]
  return prefix_groups("attention", cases)
