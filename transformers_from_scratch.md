# Transformers From Scratch

## Introduction
Transformers are one of the most influential neural network architectures in modern machine learning. They power large language models, translation systems, image generators, and more. Unlike earlier sequence models such as RNNs or LSTMs, transformers rely entirely on **attention mechanisms**, allowing them to model long-range dependencies efficiently and in parallel.

In this notebook, we will build a **complete, minimal transformer** step by step. The goal is not to optimize performance, but to deeply understand *how transformers work* by implementing each component ourselves.

By the end, you will:
- Understand self-attention intuitively and mathematically
- Implement scaled dot-product attention
- Build multi-head attention
- Construct transformer encoder blocks
- Train a tiny transformer on toy data

---

## Prerequisites
We assume basic familiarity with:
- Python
- PyTorch tensors and autograd
- Linear algebra fundamentals (vectors, matrices, dot products)

---

## Imports and Setup

We start with standard PyTorch imports. Everything in this notebook is self-contained.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
```

---

## Why Attention?

Sequence models must answer a fundamental question:

> *Which parts of the input should the model focus on when producing a representation?*

RNNs process tokens sequentially, which makes long-range dependencies difficult to learn. Attention solves this by allowing **every token to directly look at every other token** and decide what matters.

This is the core idea behind transformers.

---

## Scaled Dot-Product Attention

### Intuition
Each token produces three vectors:
- **Query (Q)**: what am I looking for?
- **Key (K)**: what do I contain?
- **Value (V)**: what information do I provide?

Attention scores are computed by comparing queries with keys. These scores determine how much of each value to mix into the output.

### Formula

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The scaling factor \( \sqrt{d_k} \) stabilizes gradients for large dimensions.

---

## Implementing Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

Let's test it with dummy data.

```python
batch, seq_len, d_model = 2, 4, 8
Q = torch.randn(batch, seq_len, d_model)
K = torch.randn(batch, seq_len, d_model)
V = torch.randn(batch, seq_len, d_model)

output = scaled_dot_product_attention(Q, K, V)
output.shape
```

---

## Self-Attention Layer

Self-attention means **Q, K, and V all come from the same input embeddings**. We learn linear projections that map embeddings into Q, K, and V spaces.

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        attn = scaled_dot_product_attention(Q, K, V)
        return self.out(attn)
```

---

## Multi-Head Attention

### Why Multiple Heads?
Instead of a single attention operation, transformers use **multiple attention heads**. Each head can focus on different types of relationships (syntax, semantics, position, etc.).

Each head works in a lower-dimensional space, and their outputs are concatenated.

---

## Implementing Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        attn = scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)
```

---

## Position-Wise Feedforward Network

Attention mixes information across tokens, but we still need **non-linear transformations applied independently to each position**.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
```

---

## Transformer Encoder Block

Each encoder block consists of:
1. Multi-head self-attention + residual connection
2. Feedforward network + residual connection
3. Layer normalization

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x
```

---

## Positional Encoding

Because transformers have no recurrence or convolution, we must inject **position information** explicitly.

We use sinusoidal positional encodings.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## Full Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## Testing the Transformer

```python
vocab_size = 50
model = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=32,
    num_heads=4,
    d_ff=64,
    num_layers=2
)

x = torch.randint(0, vocab_size, (2, 10))
out = model(x)
out.shape
```

---

## Conclusion

You have built a transformer encoder completely from scratch. While real-world models add optimizations like masking, dropout, and massive scale, the core ideas remain exactly what you've implemented here.

Transformers succeed because they:
- Model global dependencies efficiently
- Scale extremely well with data and compute
- Are simple, modular, and expressive

From here, you can extend this notebook to:
- Add causal masking
- Build a decoder
- Train on real language data
- Implement a full GPT-style model

Happy experimenting 🚀

