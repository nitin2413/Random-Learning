"""
miniGPT — built from scratch
A character-level GPT trained on Shakespeare.

Run:  python minigpt.py
Expected training time: ~5 min CPU, ~1 min GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import time
import math

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — HYPERPARAMETERS
# All config in one place. Change these to experiment.
# ═══════════════════════════════════════════════════════════════

# Data
BATCH_SIZE    = 32       # sequences processed in parallel
BLOCK_SIZE    = 128      # context window (max tokens the model sees)

# Model
D_MODEL       = 192      # embedding dimension (d_model)
N_HEADS       = 6        # number of attention heads (d_model must be divisible by N_HEADS)
N_LAYERS      = 4        # number of GPT blocks stacked
D_FF          = D_MODEL * 4  # FFN inner dimension (always 4x d_model)
DROPOUT       = 0.1      # dropout probability

# Training
MAX_ITERS     = 3000     # total training steps
EVAL_INTERVAL = 300      # evaluate loss every N steps
EVAL_ITERS    = 50       # how many batches to average for eval loss
LR            = 3e-4     # learning rate (AdamW)

# Generation
GEN_TOKENS    = 500      # how many tokens to generate at the end
TEMPERATURE   = 0.8      # higher = more random, lower = more focused
TOP_K         = 40       # only sample from top-k most likely tokens

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — DATA
# Download Shakespeare, build vocabulary, encode text
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  PHASE 1 — Loading data")
print("═"*55)

DATA_PATH = "shakespeare.txt"
if not os.path.exists(DATA_PATH):
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, DATA_PATH)
    print("Downloaded.")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size    : {len(text):,} characters")
print(f"Sample          : {repr(text[:80])}")

# ── Character-level tokenizer ─────────────────────────────────
# Vocabulary = every unique character in the text
# This is the simplest possible tokenizer — no BPE, no subwords
# GPT-2 uses BPE with 50,257 tokens; we use chars with ~65 tokens

chars   = sorted(set(text))         # all unique characters
VOCAB_SIZE = len(chars)

# Lookup tables: char ↔ integer
stoi = {ch: i for i, ch in enumerate(chars)}  # string to index
itos = {i: ch for i, ch in enumerate(chars)}  # index to string

encode = lambda s: [stoi[c] for c in s]           # "hello" → [46, 43, 50, 50, 53]
decode = lambda l: ''.join([itos[i] for i in l])   # [46,43,...] → "hello"

print(f"Vocabulary size : {VOCAB_SIZE} characters")
print(f"Vocabulary      : {repr(''.join(chars))}")

# ── Encode full dataset ───────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded shape   : {data.shape}")

# ── Train / val split (90/10) ─────────────────────────────────
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train tokens    : {len(train_data):,}")
print(f"Val tokens      : {len(val_data):,}")


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — DATASET (sliding window batching)
# ═══════════════════════════════════════════════════════════════

def get_batch(split):
    """
    Returns a random batch of (inputs, targets).

    For each sequence of length BLOCK_SIZE:
      input:  tokens[0 : BLOCK_SIZE]
      target: tokens[1 : BLOCK_SIZE+1]   ← shifted right by 1

    Example with BLOCK_SIZE=5:
      input:  [T, h, e,  ,  c]
      target: [h, e,  ,  c, a]

    At position i, the model sees input[0..i] and must predict target[i].
    This gives BLOCK_SIZE training examples per sequence.
    """
    data_split = train_data if split == 'train' else val_data

    # Sample BATCH_SIZE random starting positions
    ix     = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))

    # Stack into tensors
    x = torch.stack([data_split[i:i+BLOCK_SIZE]   for i in ix])  # (B, T)
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])  # (B, T)

    return x.to(DEVICE), y.to(DEVICE)


# Sanity check
xb, yb = get_batch('train')
print(f"\nBatch shapes — x: {xb.shape}, y: {yb.shape}")  # (32, 128), (32, 128)
print(f"First input  (decoded): {repr(decode(xb[0][:20].tolist()))}")
print(f"First target (decoded): {repr(decode(yb[0][:20].tolist()))}")


# ═══════════════════════════════════════════════════════════════
# PHASE 4 — MODEL
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  PHASE 2 — Building model")
print("═"*55)

# ── 4a. Causal Self-Attention ─────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    'Causal' = each position can only attend to itself and earlier positions.
    This is achieved via a lower-triangular mask (torch.tril).

    All heads are computed in parallel by reshaping — faster than looping.
    """

    def __init__(self):
        super().__init__()

        assert D_MODEL % N_HEADS == 0, "d_model must be divisible by n_heads"
        self.d_head = D_MODEL // N_HEADS  # dimension per head (d_k)

        # Single linear layer that projects to Q, K, V for ALL heads at once
        # Output size = 3 * D_MODEL  (Q + K + V concatenated)
        self.qkv_proj = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)

        # Output projection W^O — combines all heads back to d_model
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

        # Causal mask — register as buffer (not a parameter, but moves with .to(device))
        # tril: lower triangular matrix of ones
        # shape: (1, 1, BLOCK_SIZE, BLOCK_SIZE) — broadcasts over batch and heads
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, d_model

        # ── Step 1: compute Q, K, V for all heads in one matmul
        qkv = self.qkv_proj(x)           # (B, T, 3*D_MODEL)
        Q, K, V = qkv.split(D_MODEL, dim=2)  # each: (B, T, D_MODEL)

        # ── Step 2: reshape to (B, n_heads, T, d_head) for parallel computation
        # .view() reshapes without copying data
        # .transpose() swaps dims so heads are dim 1 (easier for matmul)
        Q = Q.view(B, T, N_HEADS, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = K.view(B, T, N_HEADS, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        V = V.view(B, T, N_HEADS, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        # ── Step 3: scaled dot-product attention
        # (B,H,T,dh) @ (B,H,dh,T) → (B,H,T,T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # Apply causal mask — future positions get -inf → softmax → 0
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        weights = self.attn_dropout(weights)

        # (B,H,T,T) @ (B,H,T,dh) → (B,H,T,dh)
        out = weights @ V

        # ── Step 4: reassemble heads
        # transpose back, then .contiguous() + .view() to merge head dim
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)  # (B, T, D_MODEL)

        # ── Step 5: output projection
        out = self.resid_dropout(self.out_proj(out))

        return out  # (B, T, D_MODEL)


# ── 4b. Feed-Forward Network ──────────────────────────────────
class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation.

    Note: original paper used ReLU. GPT-2 and most modern models use GELU.
    GELU is smoother than ReLU — works better in practice for language models.

    Expand: d_model → d_ff (4x wider)
    Contract: d_ff → d_model
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),                    # smoother than ReLU, used in GPT-2+
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, D_MODEL) → (B, T, D_MODEL)


# ── 4c. GPT Block ─────────────────────────────────────────────
class GPTBlock(nn.Module):
    """
    One transformer decoder block:
      LayerNorm → Attention → residual
      LayerNorm → FFN → residual

    Note: 'Pre-norm' formulation (norm BEFORE sublayer).
    Original paper did 'Post-norm' (norm AFTER).
    Pre-norm trains more stably — used in GPT-2 and almost everything since.
    """

    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.ff   = FeedForward()

    def forward(self, x):
        # Residual connection: x + sublayer(layernorm(x))
        x = x + self.attn(self.ln1(x))  # attention branch
        x = x + self.ff(self.ln2(x))    # FFN branch
        return x


# ── 4d. Full GPT Model ────────────────────────────────────────
class MiniGPT(nn.Module):
    """
    Full GPT model:
      token embedding + position embedding
      → N GPT blocks
      → final LayerNorm
      → linear head (d_model → vocab_size)
      → logits (not softmax — CrossEntropyLoss handles that)
    """

    def __init__(self):
        super().__init__()

        # Token embedding table: vocab_size × d_model
        # Each token index maps to a learned d_model-dimensional vector
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # Position embedding table: BLOCK_SIZE × d_model
        # Learned (not sinusoidal) — GPT-2 style
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, D_MODEL)

        self.drop      = nn.Dropout(DROPOUT)

        # Stack of N GPT blocks
        self.blocks    = nn.Sequential(*[GPTBlock() for _ in range(N_LAYERS)])

        # Final LayerNorm before the output head
        self.ln_final  = nn.LayerNorm(D_MODEL)

        # Output head: projects d_model → vocab_size (logits over vocabulary)
        # bias=False is standard in modern GPTs
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Weight tying: share weights between token embedding and output head
        # token_emb.weight shape: (vocab_size, d_model)
        # head.weight shape:      (vocab_size, d_model)
        # They represent the same information from opposite directions
        # Reduces parameters and often improves performance
        self.head.weight = self.token_emb.weight

        # Initialize weights properly
        self.apply(self._init_weights)

        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        """
        Standard GPT weight initialization.
        Linear and Embedding: normal distribution with std=0.02
        Biases: zero
        LayerNorm: weight=1, bias=0 (standard)
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) — token indices
        targets: (B, T) — ground truth next tokens (None at inference)

        Returns:
          logits: (B, T, vocab_size)
          loss:   scalar cross-entropy (None if targets not provided)
        """
        B, T = idx.shape
        assert T <= BLOCK_SIZE, f"Sequence length {T} exceeds BLOCK_SIZE {BLOCK_SIZE}"

        # ── Token + position embeddings
        tok_emb = self.token_emb(idx)                          # (B, T, D_MODEL)
        pos     = torch.arange(T, device=DEVICE)               # [0, 1, 2, ..., T-1]
        pos_emb = self.pos_emb(pos)                            # (T, D_MODEL)

        # Addition broadcasts pos_emb across batch dimension
        x = self.drop(tok_emb + pos_emb)                      # (B, T, D_MODEL)

        # ── N transformer blocks
        x = self.blocks(x)                                     # (B, T, D_MODEL)

        # ── Final norm + output head
        x      = self.ln_final(x)                             # (B, T, D_MODEL)
        logits = self.head(x)                                  # (B, T, VOCAB_SIZE)

        # ── Loss (only computed during training)
        loss = None
        if targets is not None:
            # Flatten (B, T, V) → (B*T, V) and (B, T) → (B*T,)
            # CrossEntropyLoss expects (N, C) and (N,)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation.
        Given a starting context idx (B, T), generate max_new_tokens more.

        temperature: controls randomness
          → 1.0: sample from distribution as-is
          → <1.0: more focused/repetitive (sharper distribution)
          → >1.0: more random/creative (flatter distribution)

        top_k: only sample from the k most probable tokens
          → reduces chance of picking very unlikely tokens
        """
        for _ in range(max_new_tokens):

            # Crop to BLOCK_SIZE if sequence is too long
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]

            # Forward pass — no targets, so no loss computed
            logits, _ = self(idx_cond)

            # Take logits at the LAST position (the next-token prediction)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature — divide logits before softmax
            logits = logits / temperature

            # Top-k filtering: zero out all logits except the top k
            if top_k is not None:
                # Find the kth largest value
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set everything below the kth value to -inf
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample one token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx


# ── Instantiate model ─────────────────────────────────────────
model = MiniGPT().to(DEVICE)

# Verify forward pass works before training
with torch.no_grad():
    test_x, test_y = get_batch('train')
    test_logits, test_loss = model(test_x, test_y)
    print(f"Logits shape    : {test_logits.shape}")   # (32, 128, 65)
    print(f"Initial loss    : {test_loss.item():.4f}")
    # Expected initial loss ≈ ln(vocab_size) ≈ ln(65) ≈ 4.17
    # If model is random, all tokens equally likely → entropy = log(vocab_size)
    print(f"Expected loss   : {math.log(VOCAB_SIZE):.4f}  (random baseline = ln({VOCAB_SIZE}))")


# ═══════════════════════════════════════════════════════════════
# PHASE 5 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  PHASE 3 — Training")
print("═"*55)

# AdamW — Adam with weight decay
# Weight decay acts as L2 regularization — prevents weights from growing too large
# lr=3e-4 is the standard starting point for transformers (Karpathy's rule of thumb)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)


@torch.no_grad()
def estimate_loss():
    """
    Evaluate average loss on train and val sets.
    Uses EVAL_ITERS batches to reduce noise.
    .eval() disables dropout for a fair estimate.
    """
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ── Training loop ─────────────────────────────────────────────
start_time = time.time()

for step in range(MAX_ITERS):

    # Evaluate and print loss every EVAL_INTERVAL steps
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"step {step:4d}/{MAX_ITERS} | "
              f"train loss: {losses['train']:.4f} | "
              f"val loss: {losses['val']:.4f} | "
              f"time: {elapsed:.0f}s")

    # ── Forward pass
    xb, yb    = get_batch('train')
    logits, loss = model(xb, yb)

    # ── Backward pass
    optimizer.zero_grad(set_to_none=True)  # clear gradients from previous step
    loss.backward()                         # compute gradients

    # Gradient clipping — prevents gradient explosion in deep networks
    # If gradient norm > 1.0, scale all gradients down proportionally
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()                        # update weights

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.0f}s")


# ═══════════════════════════════════════════════════════════════
# PHASE 6 — GENERATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  PHASE 4 — Text Generation")
print("═"*55)

model.eval()

def generate_text(prompt="", max_tokens=GEN_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    else:
        # Start from newline token if no prompt
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    generated = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    return decode(generated[0].tolist())


# ── Generation 1: from empty context ──────────────────────────
print("\n[Generation 1] Temperature=0.8, top_k=40")
print("─" * 50)
print(generate_text())

# ── Generation 2: with a prompt ───────────────────────────────
print("\n[Generation 2] Prompt: 'ROMEO:'")
print("─" * 50)
print(generate_text(prompt="ROMEO:", max_tokens=300))

# ── Generation 3: temperature experiment ──────────────────────
print("\n[Generation 3] Low temperature (0.3) — more focused")
print("─" * 50)
print(generate_text(max_tokens=200, temperature=0.3))

print("\n[Generation 4] High temperature (1.2) — more random")
print("─" * 50)
print(generate_text(max_tokens=200, temperature=1.2))


# ═══════════════════════════════════════════════════════════════
# PHASE 7 — WHAT YOU BUILT (model summary)
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  PHASE 5 — Model Summary")
print("═"*55)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nArchitecture:")
print(f"  Vocabulary size   : {VOCAB_SIZE}")
print(f"  Context window    : {BLOCK_SIZE} tokens")
print(f"  d_model           : {D_MODEL}")
print(f"  Attention heads   : {N_HEADS}")
print(f"  d_head (per head) : {D_MODEL // N_HEADS}")
print(f"  GPT blocks        : {N_LAYERS}")
print(f"  d_ff              : {D_FF}")

print(f"\nParameters:")
print(f"  Token embedding   : {VOCAB_SIZE * D_MODEL:>10,}  ({VOCAB_SIZE} × {D_MODEL})")
print(f"  Position embedding: {BLOCK_SIZE * D_MODEL:>10,}  ({BLOCK_SIZE} × {D_MODEL})")
for name, p in model.named_parameters():
    if 'blocks.0' in name:
        print(f"  [Block 0] {name:<30}: {p.numel():>8,}")
print(f"  Total (all layers): {total_params:>10,}")
print(f"  Trainable params  : {trainable:>10,}")

print(f"\nTraining:")
print(f"  Steps             : {MAX_ITERS}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Context window    : {BLOCK_SIZE}")
print(f"  Tokens per step   : {BATCH_SIZE * BLOCK_SIZE:,}")
print(f"  Total tokens seen : {MAX_ITERS * BATCH_SIZE * BLOCK_SIZE:,}")
print(f"  Optimizer         : AdamW (lr={LR}, wd=0.1)")
print(f"  Device            : {DEVICE}")
print(f"  Training time     : {total_time:.0f}s")

print("\n" + "═"*55)
print("  miniGPT complete ✅")
print("═"*55)


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS TO TRY AFTER IT RUNS
# ═══════════════════════════════════════════════════════════════
"""
EXPERIMENT 1 — Scale up the model
  D_MODEL  = 384
  N_HEADS  = 6
  N_LAYERS = 6
  MAX_ITERS = 5000
  → Loss should drop to ~1.4

EXPERIMENT 2 — Change context window
  BLOCK_SIZE = 256
  → Longer dependencies, needs more training

EXPERIMENT 3 — Temperature sweep
  generate_text(temperature=0.1)  # very repetitive, "safe"
  generate_text(temperature=0.5)  # balanced
  generate_text(temperature=1.0)  # raw distribution
  generate_text(temperature=1.5)  # chaotic

EXPERIMENT 4 — Train on your own text
  Replace shakespeare.txt with any .txt file
  Works on: code, poetry, wikipedia dumps, tweets, anything

EXPERIMENT 5 — Overfit on tiny data
  text = text[:10000]   # only first 10k chars
  MAX_ITERS = 2000
  → Train loss should hit ~0.3 (memorization)
  → Val loss stays high (no generalization)
  → This is what overfitting looks like
"""