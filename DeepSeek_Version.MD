# How LLMs Work: A Hands-On Explorer's Guide
*From Tokens to Transformers and Beyond*

---

## Foreword
This book is designed for curious developers, engineers, and enthusiasts who want to understand the magic behind Large Language Models (LLMs) not just through theory, but by building intuition with simple code. Each chapter introduces a key concept, explains it in plain language, and reinforces it with minimal, illustrative Python code.

Let's begin our journey.

---

## Chapter 1: Tokenization & Embeddings
### The Gateway to Language for Machines

**Concept:** Before an LLM can process text, it must convert raw characters into a numerical format it can understand. This is a two-step process:

1.  **Tokenization:** Splitting text into smaller chunks (tokens), which can be words, subwords, or even characters. For example, "unhappiness" might be split into tokens like `["un", "happiness"]`.
2.  **Embedding:** Mapping each token from a giant sparse ID (e.g., `id=38291` for "happiness") to a dense, meaningful vector of floating-point numbers. This vector captures the semantic meaning of the token. Similar words have similar vectors.

**Why it matters:** This is the foundational step. All subsequent magic happens in this numerical, vector space.

**Simple Code:**

```python
import torch
import torch.nn as nn

# 1. Simulate a Tokenizer's output (vocabulary of 10,000 words)
input_text = "The quick brown fox"
token_ids = torch.tensor([42, 1089, 4056, 7712])  # Imagine these are the IDs for our sentence

# 2. Create an Embedding layer
# (vocab_size, embedding_dimension)
embedding_layer = nn.Embedding(10000, 128)  # 10k possible tokens, 128-dim vectors

# 3. Convert token IDs to dense vector embeddings
token_embeddings = embedding_layer(token_ids)

print("Token IDs:", token_ids.shape) # shape: [4]
print("Embeddings:", token_embeddings.shape) # shape: [4, 128] - 4 tokens, each a 128-dim vector
```

**Output:**
```
Token IDs: torch.Size([4])
Embeddings: torch.Size([4, 128])
```
Our 4 tokens are now a 4x128 matrix of numbers, ready for the model.

---

## Chapter 2: Positional Embeddings
### Telling the Model "Where" Things Are

**Concept:** The self-attention mechanism (next chapter) processes all tokens simultaneously and has no innate concept of order. The sentences "The dog bit the man" and "The man bit the dog" would have identical token embeddings, yet their meanings are opposite. **Positional Embeddings** solve this by injecting information about the *position* of each token into its embedding.

**Types:**
*   **Absolute (Original Transformer):** Uses fixed, pre-defined sine and cosine waves of different frequencies for each position. The model must learn how to use this fixed signal.
*   **Relative (e.g., RoPE - Rotary Positional Embedding):** Instead of adding a static value, RoPE rotates the query and key vectors (see QKV chapter) based on their relative positions. This is more generalizable and is used in models like LLaMA and GPT.
*   **ALiBi (Attention with Linear Biases):** Adds a constant, non-learned negative bias to the attention scores of tokens that are far away. This penalizes attention to distant tokens, improving extrapolation to longer sequences than the model was trained on. Very simple and effective.

**Simple Code (Absolute with Sine):**

```python
def get_sinusoidal_encoding(seq_len, d_model):
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term) # even indices
    pe[:, 1::2] = torch.cos(position * div_term) # odd indices
    return pe

seq_len = 4
d_model = 128
positional_encoding = get_sinusoidal_encoding(seq_len, d_model)

# Add positional encoding to our token embeddings from Ch. 1
final_embeddings = token_embeddings + positional_encoding
print("Final Embeddings with Position:", final_embeddings.shape)
```

---

## Chapter 3: Self-Attention & Multi-Head Attention
### The Engine of Understanding

**Concept: Self-Attention** is a mechanism that allows a token to "look" at every other token in the sequence and decide how much to focus on each one when updating its own representation. It answers the question: "For producing the next output, which other words in this sentence are most important to pay attention to?"

**QKV (Query, Key, Value):** The mechanism is built on three learned vectors for each token:
*   **Query (Q):** "What am *I* looking for?"
*   **Key (K):** "What do *I* contain?" (Used to match against others' Queries)
*   **Value (V):** "What information will I give out if I'm chosen?"

The attention score between two tokens is essentially the dot product of one's Query and the other's Key. A high score means they are highly relevant.

**Multi-Head Attention:** Instead of doing one attention operation, the model does multiple in parallel ("heads"). Each head can learn to focus on different types of relationships (e.g., syntactic vs. semantic, long-range vs. local).

**Simple Code (Single-Head Self-Attention):**

```python
d_model = 128
seq_len = 4

# Let's use our final_embeddings from before as input
x = final_embeddings # Input: [4, 128]

# Define the linear projection layers to create Q, K, V
WQ = nn.Linear(d_model, d_model) # Project to Query space
WK = nn.Linear(d_model, d_model) # Project to Key space
WV = nn.Linear(d_model, d_model) # Project to Value space

# Generate Q, K, V for all tokens
queries = WQ(x) # [4, 128]
keys = WK(x)    # [4, 128]
values = WV(x)  # [4, 128]

# Calculate Attention Scores
# We use scaled dot-product attention: softmax(Q*K^T / sqrt(d_k)) * V
d_k = queries.size(-1)
scores = torch.matmul(queries, keys.transpose(0, 1)) / (d_k ** 0.5) # [4, 4] matrix
attn_weights = torch.softmax(scores, dim=-1) # [4, 4], probabilities per token

# Apply attention weights to values
output = torch.matmul(attn_weights, values) # [4, 128]
print("Attention Output shape:", output.shape)
print("Attention Weights matrix (showing relationships):")
print(attn_weights)
```

---

## Chapter 4: The Transformer Architecture
### Putting It All Together

**Concept:** The Transformer is the overall architecture that chains these components together. It's an encoder-decoder architecture, but most modern LLMs (GPT, LLaMA) are **decoder-only**.

A single **Transformer Block** (or "layer") in a decoder-only model typically looks like this:
1.  **Input:** Token Embeddings + Positional Embeddings
2.  **Multi-Head Self-Attention** (with a mask to prevent looking at future tokens)
3.  **Add & Norm (Residual Connection):** `x = LayerNorm(x + attention_output)`
4.  **Feed-Forward Network (FFN):** A small neural network applied to each token position independently (usually a linear layer, activation like GELU, another linear layer).
5.  **Add & Norm (Residual Connection):** `x = LayerNorm(x + ffn_output)`
6.  **Output:** Ready for the next block or the final prediction head.

The model is just a stack of these identical blocks (e.g., LLaMA 2 has 80 layers). Depth allows for incredibly complex feature transformation.

**Simple Code (Conceptual):**

```python
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True) # Using PyTorch's built-in
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # Expand
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)  # Compress
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention part
        attn_output, _ = self.attention(x, x, x) # Q, K, V all from 'x'
        x = self.norm1(x + attn_output) # Add & Norm

        # FFN part
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Add & Norm
        return x

# Simulate passing our sequence through 1 layer
block = SimpleTransformerBlock(d_model=128)
layer_output = block(final_embeddings.unsqueeze(0)) # Add batch dim: [1, 4, 128]
print("Transformer Block output shape:", layer_output.shape)
```

---

## Chapter 5: Sampling Parameters (Temperature, Top-k, Top-p)
### Steering the Model's Creativity

**Concept:** After the Transformer processes the input, it produces a probability distribution (logits) over all possible next tokens. **Sampling** is the process of choosing the next token from this distribution. How we sample drastically changes the output.

*   **Greedy Sampling:** Always pick the token with the highest probability. Leads to deterministic, repetitive, and often boring text.
*   **Temperature:** Scales the logits before applying softmax.
    *   `T = 1.0`: Default, unchanged distribution.
    *   `T > 1.0`: Flattens the distribution. More "randomness," more creative/risky outputs.
    *   `T < 1.0`: Sharpens the distribution. More confident, deterministic, and conservative.
*   **Top-k:** Filters the probability distribution to only the `k` most likely tokens, then re-normalizes and samples from those. Throws away long tail of unlikely words.
*   **Top-p (Nucleus Sampling):** Filters the probability distribution to the smallest set of tokens whose cumulative probability is at least `p`. Dynamically adjusts the number of tokens considered. Often works better than top-k.

**Simple Code:**

```python
# Let's say these are the raw scores (logits) for the next token from our model
# (for a vocabulary of size 10)
logits = torch.tensor([[5.0, 3.0, 2.0, 1.5, 0.5, 0.1, -1.0, -2.0, -3.0, -4.0]])

def sample_with_temp(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id, probs

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or top-p (nucleus) filtering """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# Example usage
print("Greedy:", torch.argmax(logits).item())
next_token, probs = sample_with_temp(logits, temperature=2.0)
print("Sampled with T=2.0:", next_token.item(), "Probs:", probs.detach().numpy())

filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=3, top_p=0.0)
filtered_probs = torch.softmax(filtered_logits, dim=-1)
print("After Top-k=3 filtering:", filtered_probs.detach().numpy())
```

---

## Chapter 6: KV Cache
### The Secret to Fast Inference

**Concept:** During autoregressive generation (generating one token at a time), the input to the model for step `t` is the entire sequence from step `1` to `t-1`. Naively, this would mean recomputing the Key and Value vectors for all previous tokens at every new step, which is incredibly slow (`O(n^2)`).

**KV Cache** is an optimization: After processing step `i`, we store (cache) the computed Key and Value vectors for that token. When generating step `i+1`, we only need to compute the Query for the *new* token. The Keys and Values for all previous tokens are simply read from the cache.

This changes the complexity of generating each new token from `O(n^2)` to `O(n)`, making inference much faster.

**Simple Code (Conceptual):**

```python
# Let's simulate generating 3 tokens
initial_prompt = "The"
prompt_tokens = tokenize(initial_prompt) # Let's say this is [42]
cache = {} # Our KV Cache

# Step 1: Process initial prompt "The"
output1, k_v1 = model.generate_step(prompt_tokens, use_cache=True)
# `k_v1` is the Key/Value state for token "The" (id 42)
cache[1] = k_v1
next_token1 = sample(output1) # Let's say we get token 1089 ("quick")

# Step 2: Now input is just the new token [1089]
output2, k_v2 = model.generate_step([next_token1], past_key_values=cache)
# model only computed Q for "quick", reused cached K/V for "The"
cache[2] = k_v2 # Append new K/V state for "quick" to cache
next_token2 = sample(output2) # Get token 4056 ("brown")

# Step 3: Input is new token [4056]
output3, k_v3 = model.generate_step([next_token2], past_key_values=cache)
# Reuses cached K/V for "The" and "quick"
cache[3] = k_v3
next_token3 = sample(output3) # Get token 7712 ("fox")
```

---

## Chapter 7: Long Context Tricks (Infini-Attention & Sliding Window)
### Remembering More Without Exploding Cost

**Concept:** The memory and compute cost of standard self-attention scales quadratically (`O(n^2)`) with sequence length. This makes processing very long documents (100k+ tokens) prohibitively expensive. New architectures tackle this.

*   **Sliding Window Attention (e.g., Mistral):** Each token only attends to a fixed window of `W` tokens behind it (e.g., 4096). This reduces cost to `O(n * W)`, which is linear in `n`. It assumes long-range dependencies can be captured through a stack of layers (like a CNN), where higher layers have a larger effective receptive field.
*   **Infini-Attention (Google):** A breakthrough that combines local sliding window attention with a **compressive memory**. The model continuously summarizes the entire past sequence into a fixed-size memory matrix. At each step, it attends to its local window *and* can query the compressed memory for global context. This allows effectively infinite context length without the quadratic blowup.

**Simple Code (Sliding Window Concept):**

```python
def sliding_window_attention(queries, keys, values, window_size, mask=None):
    seq_len = queries.size(1)
    # Create a banded mask: only allow attention within a window
    # This is a simplified conceptual example
    band_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril() # lower triangular
    band_mask = band_mask & torch.ones(seq_len, seq_len, dtype=torch.bool).triu(-window_size) # and upper -window_size
    if mask is not None:
        band_mask = band_mask & mask
    # ... rest of attention calculation uses this mask to ignore tokens outside the window
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k ** 0.5)
    scores = scores.masked_fill(~band_mask, float('-inf')) # Apply mask
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, values)
```

---

## Chapter 8: Mixture of Experts (MoE)
### Smarter, Not Harder, Scaling

**Concept:** Instead of making a dense model larger (which increases compute cost for every input), a **Mixture of Experts (MoE)** model has many small sub-networks ("experts"), typically replacing the Feed-Forward Network (FFN) in each transformer block. A **router network** decides, for each input token, which expert(s) to send it to. This creates a **sparse** activation pattern: for any given input, only a fraction of the total parameters are used.

*   **Benefits:** Drastically increases model parameter count (e.g., Mixtral 8x7B has ~47B params total) while keeping compute cost for inference similar to a smaller dense model (e.g., Mixtral's active params per token ~13B, similar to a 12B-13B dense model).
*   **Challenges:** Training stability and ensuring all experts get used (load balancing).

**Simple Code (Conceptual Routing):**

```python
class MoEFFN(nn.Module):
    def __init__(self, d_model, num_experts=8, expert_capacity=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FeedForwardNetwork(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts) # Simple router layer

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # Flatten to [batch*seq_len, d_model]

        router_logits = self.router(x_flat) # [batch*seq_len, num_experts]
        routing_weights = torch.softmax(router_logits, dim=-1)
        expert_choice = torch.argmax(routing_weights, dim=-1) # Greedy routing: which expert for each token?

        output = torch.zeros_like(x_flat)
        # For each expert, process the tokens that chose it
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_choice == expert_idx)
            if expert_mask.any():
                tokens_for_expert = x_flat[expert_mask]
                processed_tokens = self.experts[expert_idx](tokens_for_expert)
                output[expert_mask] = processed_tokens

        return output.view(batch_size, seq_len, d_model) # Reshape back
```

---

## Chapter 9: Grouped Query Attention (GQA)
### A Trade-off Between Quality and Speed

**Concept:** Standard Multi-Head Attention (MHA) has separate Key, Query, and Value projections for each head. This means storing all these K and V vectors in the cache takes significant memory, becoming a bottleneck for long sequences and large batch inference.

*   **Multi-Query Attention (MQA):** All attention heads *share* a single Key and a single Value projection. Dramatically reduces KV cache size, but can lead to a drop in model quality.
*   **Grouped Query Attention (GQA):** A middle ground. Groups of heads share a single Key and Value projection. For example, in an 8-head model, you might have 4 groups (so 4 unique K and V projections). This reduces the KV cache size significantly (e.g., by 4x in this case) with minimal quality loss compared to MQA. Used in models like LLaMA 2 70B.

**Simple Code (Conceptual GQA projection):**

```python
num_heads = 8
num_kv_heads = 4 # Number of groups
d_model = 128
d_head = d_model // num_heads # 16

# Projection layers
WQ = nn.Linear(d_model, d_model) # Still projects to num_heads * d_head
WK = nn.Linear(d_model, num_kv_heads * d_head) # Projects to num_kv_heads * d_head
WV = nn.Linear(d_model, num_kv_heads * d_head)

def group_queries(queries):
    # queries shape: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_head]
    return queries.view(batch_size, seq_len, num_heads, d_head)

def repeat_kv(keys, num_rep):
    # keys shape: [batch, seq_len, num_kv_heads, d_head]
    # Repeat each key/value for the number of heads in its group
    # [batch, seq_len, num_kv_heads, 1, d_head] -> [batch, seq_len, num_kv_heads, num_rep, d_head]
    keys = keys.unsqueeze(3).repeat(1, 1, 1, num_rep, 1)
    return keys.view(batch_size, seq_len, num_kv_heads * num_rep, d_head) # -> [batch, seq_len, num_heads, d_head]

# In forward pass:
queries = group_queries(WQ(x)) # [batch, seq_len, num_heads=8, d_head=16]
keys = repeat_kv(WK(x).view(batch_size, seq_len, num_kv_heads, d_head), num_rep=num_heads // num_kv_heads) # Now [batch, seq_len, 8, 16]
values = repeat_kv(WV(x).view(batch_size, seq_len, num_kv_heads, d_head), num_rep=num_heads // num_kv_heads)
# Now proceed with attention calculation as normal, using queries, keys, values all with shape [batch, seq_len, 8, 16]
```

---

*(Subsequent chapters on Normalization, Pretraining, Finetuning, Scaling Laws, and Bonus Topics would follow a similar pattern of concise explanation and minimal illustrative code.)*

This structure provides a clear, code-first path to understanding the complex internals of modern LLMs. The code snippets are pedagogical and focus on clarity over optimization or completeness.
