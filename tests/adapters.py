from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum


class LinearModule(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
    
    def forward(self, in_features: torch.Tensor, weights: torch.Tensor):
        assert self.d_in == in_features.shape[-1]
        assert self.d_out == weights.shape[0]
        return einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out")


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    LinearLayer = LinearModule(d_in, d_out)
    return LinearLayer.forward(in_features, weights)

class EmbeddingModule(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, weights: Float[Tensor, " vocab_size d_model"], token_ids: Int[Tensor, " ..."]):
        (vocab_size, d_model) = weights.shape
        assert vocab_size == self.vocab_size
        assert d_model == self.d_model
        return weights[token_ids]

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    EmbeddingLayer = EmbeddingModule(vocab_size, d_model)
    return EmbeddingLayer.forward(weights, token_ids)

class SwigluModule(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, w1_weight: Float[Tensor, " d_ff d_model"],
                w2_weight: Float[Tensor, " d_model d_ff"],
                w3_weight: Float[Tensor, " d_ff d_model"],
                in_features: Float[Tensor, " ... d_model"]):
        assert w1_weight.shape[0] == self.d_ff
        assert w2_weight.shape == (self.d_model, self.d_ff)
        assert w3_weight.shape == (self.d_ff, self.d_model)
        assert in_features.shape[-1] == self.d_model
        W_1_x = einsum(w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
        Silu_W_1_x = W_1_x / (1 + torch.e ** -W_1_x) # (..., d_ff)
        W_3_x = einsum(w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
        res = Silu_W_1_x * W_3_x # (..., d_ff)
        return einsum(w2_weight, res, "d_model d_ff, ... d_ff -> ... d_model")

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    SwigluLayer = SwigluModule(d_model, d_ff)
    return SwigluLayer.forward(w1_weight, w2_weight, w3_weight, in_features)

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask: torch.tensor | None = None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(dim0=-1, dim1=-2)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.matmul(torch.softmax(scores, dim=-1), V)
        return attn_weights

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaled_dot_product_attention = ScaledDotProductAttention()
    return scaled_dot_product_attention.forward(Q, K, V, mask)

import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        num_heads: int,
        d_model: int, 
    ):
        # d_in = d_model
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
    def forward(
        self, 
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " .., sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        
        batch_size, seq_len, d_model = in_features.shape
        assert d_model == self.d_model
        
        d_q, d_in = q_proj_weight.shape
        assert d_in == self.d_model

        d_k, d_in = k_proj_weight.shape
        assert d_in == self.d_model
        
        d_v, d_in = v_proj_weight.shape
        assert d_in == self.d_model

        Q = torch.matmul(in_features, q_proj_weight.transpose(-2, -1)) # (.., seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, d_k // self.num_heads) # (.., seq_len, n, d_k / n)
        Q = Q.transpose(1, 2) # (..., n, seq_len, d_k / n)
        K = torch.matmul(in_features, k_proj_weight.transpose(-2, -1)) # (.., seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, d_k // self.num_heads) # (.., seq_len, n, d_k / n)
        K = K.transpose(1, 2) # (..., n, seq_len, d_k / n)
        V = torch.matmul(in_features, v_proj_weight.transpose(-2, -1)) # (.., seq_len, d_v)
        V = V.view(batch_size, seq_len, self.num_heads, d_v // self.num_heads) # (.., seq_len, n, d_v / n)
        V = V.transpose(1, 2) # (..., n, seq_len, d_v / n)

        mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
        prob = torch.nn.Softmax(-1)(torch.matmul(Q, K.transpose(-2, -1)).masked_fill(mask==1, -1e9) / torch.sqrt(torch.tensor(d_k / self.num_heads))) # (.., n, seq_len, seq_len)
        atten = torch.matmul(prob, V) # (.., n, seq_len, d_v / n)
        atten = atten.transpose(1, 2).reshape(batch_size, seq_len, d_v)
        atten = atten.view(batch_size, seq_len, d_v) # (.., seq_len, d_v)
        mha = torch.matmul(atten, o_proj_weight.transpose(-2, -1))
        return mha     


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA = MultiHeadAttention(num_heads, d_model)
    return MHA.forward(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)

class MultiHeadAttentionRope(nn.Module):
    def __init__(
        self, 
        num_heads: int,
        d_model: int,
        max_seq_len: int,
        theta: float,
    ):
        # d_in = d_model
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    
    def forward(
        self, 
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " .., sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        
        batch_size, seq_len, d_model = in_features.shape
        assert d_model == self.d_model
        
        d_q, d_in = q_proj_weight.shape
        assert d_in == self.d_model

        d_k, d_in = k_proj_weight.shape
        assert d_in == self.d_model
        
        d_v, d_in = v_proj_weight.shape
        assert d_in == self.d_model

        Q = torch.matmul(in_features, q_proj_weight.transpose(-2, -1)) # (.., seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, d_k // self.num_heads) # (.., seq_len, n, d_k / n)
        Q = Q.transpose(1, 2) # (..., n, seq_len, d_k / n)
        # Q: (batch_size, seq_len, d_k / n)  token_positions: (batch_size, seq_len)
        Q = self.rope.forward(Q, token_positions)
        
        K = torch.matmul(in_features, k_proj_weight.transpose(-2, -1)) # (.., seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, d_k // self.num_heads) # (.., seq_len, n, d_k / n)
        K = K.transpose(1, 2) # (..., n, seq_len, d_k / n)
        # K: (batch_size, seq_len, d_k / n)  token_positions: (batch_size, seq_len)
        K = self.rope.forward(K, token_positions)
        
        V = torch.matmul(in_features, v_proj_weight.transpose(-2, -1)) # (.., seq_len, d_v)
        V = V.view(batch_size, seq_len, self.num_heads, d_v // self.num_heads) # (.., seq_len, n, d_v / n)
        V = V.transpose(1, 2) # (..., n, seq_len, d_v / n)

        mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
        prob = torch.nn.Softmax(-1)(torch.matmul(Q, K.transpose(-2, -1)).masked_fill(mask==1, -1e9) / torch.sqrt(torch.tensor(d_k / self.num_heads))) # (.., n, seq_len, seq_len)
        atten = torch.matmul(prob, V) # (.., n, seq_len, d_v / n)
        atten = atten.transpose(1, 2).reshape(batch_size, seq_len, d_v)
        atten = atten.view(batch_size, seq_len, d_v) # (.., seq_len, d_v)
        mha = torch.matmul(atten, o_proj_weight.transpose(-2, -1))
        return mha  

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA_ROPE = MultiHeadAttentionRope(num_heads, d_model, max_seq_len, theta)
    return MHA_ROPE.forward(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions)

import torch.nn as nn
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError('d_k must be even')
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        freq = 1.0 / (self.theta**(torch.arange(0, d_k, 2).float() / self.d_k))

        cos_cache =  torch.outer(torch.arange(0, max_seq_len), freq) # (max_seq_len, d/2)
        sin_cache = torch.outer(torch.arange(0, max_seq_len), freq) # (max_seq_len, d/2)
        self.register_buffer('cos_cache', cos_cache.cos(), persistent=False) # (max_seq_len, d/2)
        self.register_buffer('sin_cache', sin_cache.sin(), persistent=False) # (max_seq_len, d/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_k)
        # token_positions: (batch_size, seq_len)
        cos = self.cos_cache[token_positions] # (batch_size, seq_len, d/2)
        sin = self.sin_cache[token_positions] # (batch_size, seq_len, d/2)
        even = x[..., 0::2] # [batch_size, seq_len, d_k/2]
        odd = x[..., 1::2] # [batch_size, seq_len, d_k/2]
        part_1 = cos*even - sin*odd
        part_2 = cos*odd + sin*even
        res = torch.stack([part_1, part_2], dim=-1).flatten(start_dim=-2, end_dim=-1)
        return res
        

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope.forward(in_query_or_key, token_positions)

class TransforMerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, 
                 attn_q_proj_weight: torch.Tensor,
                 attn_k_proj_weight: torch.Tensor,
                 attn_v_proj_weight: torch.Tensor,
                 attn_output_proj_weight: torch.Tensor,
                 ln1_weight: torch.Tensor,
                 ffn_w1_weight: torch.Tensor,
                 ffn_w2_weight: torch.Tensor,
                 ffn_w3_weight: torch.Tensor,
                 ln2_weight: torch.Tensor):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.attn_q_proj_weight = attn_q_proj_weight # (d_model, d_model)
        self.attn_k_proj_weight = attn_k_proj_weight # (d_model, d_model)
        self.attn_v_proj_weight = attn_v_proj_weight # (d_model, d_model)
        self.attn_output_proj_weight = attn_output_proj_weight # (d_model, d_model)
        self.ln1_weight = ln1_weight # (d_model,)
        self.ffn_w1_weight = ffn_w1_weight # (d_model, d_ff)
        self.ffn_w2_weight = ffn_w2_weight # (d_ff, d_model)
        self.ffn_w3_weight = ffn_w3_weight # (d_model, d_ff)
        self.ln2_weight = ln2_weight # (d_model,)

        self.RMS_Layer = RMSNormModule(d_model, 1e-5)
        self.MHA_ROPE_Layer = MultiHeadAttentionRope(self.num_heads, self.d_model, self.max_seq_len, self.theta)
        self.FFN_Layer = SwigluModule(self.d_model, self.d_ff)

    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]):
        assert in_features.shape[-1] == self.d_model
        token_positions = torch.arange(in_features.shape[1])

        part_1_rmsnorm = self.RMS_Layer.forward(self.ln1_weight, in_features)
        part_1_mha_rope = self.MHA_ROPE_Layer.forward(
            self.attn_q_proj_weight, 
            self.attn_k_proj_weight, 
            self.attn_v_proj_weight, 
            self.attn_output_proj_weight, 
            part_1_rmsnorm,
            token_positions)
        
        part_1_output = in_features + part_1_mha_rope

        part_2_rmsnorm = self.RMS_Layer.forward(self.ln2_weight, part_1_output)
        part_2_ffn = self.FFN_Layer.forward(self.ffn_w1_weight, self.ffn_w2_weight, self.ffn_w3_weight, part_2_rmsnorm)
        part_2_output = part_1_output + part_2_ffn

        return part_2_output




def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransforMerBlock(d_model, num_heads, d_ff, max_seq_len, theta, 
        weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"]
        , weights["attn.output_proj.weight"], weights["ln1.weight"], weights["ffn.w1.weight"],
        weights["ffn.w2.weight"], weights["ffn.w3.weight"], weights["ln2.weight"])
    return transformer_block.forward(in_features)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, weights: dict[str, Tensor],):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.EmbeddingLayer = EmbeddingModule(self.vocab_size, self.d_model)
        self.RmsNormLayer = RMSNormModule(self.d_model, 1e-5)
        self.LinearLayer = LinearModule(self.d_model, self.vocab_size)
        self.SoftMaxLayer = SoftMaxModule(-1)
    
    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        # [batch_size, seq_len, d_model]
        token_embedding = self.EmbeddingLayer.forward(self.weights["token_embeddings.weight"], in_indices)
        for num_layers in range(self.num_layers):
            transformer_block = TransforMerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.context_length,
                self.rope_theta,
                self.weights[f"layers.{num_layers}.attn.q_proj.weight"],
                self.weights[f"layers.{num_layers}.attn.k_proj.weight"],
                self.weights[f"layers.{num_layers}.attn.v_proj.weight"],
                self.weights[f"layers.{num_layers}.attn.output_proj.weight"],
                self.weights[f"layers.{num_layers}.ln1.weight"],
                self.weights[f"layers.{num_layers}.ffn.w1.weight"],
                self.weights[f"layers.{num_layers}.ffn.w2.weight"],
                self.weights[f"layers.{num_layers}.ffn.w3.weight"],
                self.weights[f"layers.{num_layers}.ln2.weight"],
            )
            token_embedding = transformer_block.forward(token_embedding)
        ffn_output = self.RmsNormLayer.forward(self.weights["ln_final.weight"], token_embedding)
        linear_output = self.LinearLayer.forward(ffn_output, self.weights["lm_head.weight"])
        return linear_output
        # return self.SoftMaxLayer.forward(linear_output)
        

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transforer_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads,
        d_ff, rope_theta, weights)
    return transforer_lm.forward(in_indices)

class RMSNormModule(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, weights: Float[Tensor, " d_model"], in_features: Float[Tensor, " ... d_model"]):
        d_model = in_features.shape[-1]
        assert d_model == self.d_model
        in_type = in_features.dtype
        in_features = in_features.to(in_type)
        RMS = torch.sqrt((in_features ** 2).sum(dim = -1, keepdim=True) / self.d_model + self.eps) # (..., 1)
        return ((in_features / RMS) * weights).to(in_type) 

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    RMSLayer = RMSNormModule(d_model, eps)
    return RMSLayer.forward(weights, in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError

class SoftMaxModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, in_features: Float[Tensor, " ..."]):
        # 假设 in_features: [batch_size, seq_len, d_k] dim = 2
        # max_num: [batch_size, seq_len, 1] 即指定的 dim 维度会变成 1
        # 由于 max 函数返回值既有tensor还有indices因此要用 [0]
        max_num = in_features.max(dim=self.dim, keepdim=True)[0] # [batch_size, seq_len, 1]
        residue = in_features - max_num # [batch_size, seq_len, d_k]
        exp_sum = torch.sum(torch.exp(residue), dim=self.dim, keepdim=True)  # exp_sum: [batch_size, seq_len, 1]
        return torch.exp(residue) / exp_sum # [batch_size, seq_len, d_k]

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    SoftMaxLayer = SoftMaxModule(dim)
    return SoftMaxLayer.forward(in_features)
    


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
