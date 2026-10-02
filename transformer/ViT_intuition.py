# reference trough claude
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmb(nn.Module):
    def __init__(self,img_size = 224, in_ch=3, patch_size = 16 ,d_model = 768):
        super().__init__()

        # ONE Conv2d replaces: "loop over patches → flatten → multiply W_E"
        # kernel=stride=16  →  no overlap, exactly one patch per output cell

        self.n_patch = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch,d_model,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        # x shape coming in:  [B, 3, 224, 224]
        x = self.proj(x)  # → [B, 768, 14, 14]  (14 = 224/16)
        x = x.flatten(2)  # → [B, 768, 196]      flatten spatial dims
        x = x.transpose(1, 2)  # → [B, 196, 768]      seq-first for transformer
        return x

    # ── why the transpose? ────────────────────────────────────────
    # Conv2d outputs [B, channels, H, W] — channels first
    # Transformer expects [B, sequence, features] — sequence first
    # flatten(2):    [B, 768, 14, 14] → [B, 768, 196]
    # transpose(1,2):[B, 768, 196]    → [B, 196, 768]  ← what we need

# ── shape check ───────────────────────────────────────────────
# pe = PatchEmb()
# img = torch.randn(2 , 3 , 224 , 224)
# x=pe(img)
# print(x.shape)

B = 2
d_models = 768
n_patch = 196

patch_token = torch.randn(B , n_patch , d_models)

# ── CLS token ─────────────────────────────────────────────────
# nn.Parameter = a tensor that IS a learnable weight (updated by backprop)
# Shape [1, 1, 768]: one token, one position, d_model features
cls_token = nn.Parameter(torch.zeros(1,1,d_models))
nn.init.trunc_normal_(cls_token,std=0.02) # don't leave it at zeros

# expand copies the CLS token for every item in the batch
cls_expand = cls_token.expand(B,-1,-1)  # [2, 1, 768]

# Prepend: CLS goes at position 0, patches follow
token = torch.cat([cls_expand,patch_token],dim=1) # → [2, 197, 768]  (196 patches + 1 CLS)

# ── Positional embedding ──────────────────────────────────────
# One learned vector per position: 197 positions × 768 dims
# Shape [1, 197, 768] — batch dim=1 so it broadcasts across B
pos_emb = nn.Parameter(torch.zeros(1,n_patch+1,d_models))
nn.init.trunc_normal_(pos_emb,std=0.02)

tokens = token + pos_emb
print(tokens.shape)

# ── What actually changed? ────────────────────────────────────
# Before pos_emb: patch at position 5 and patch at position 147
#   could be mistaken for each other (same kind of vector)
# After pos_emb:  each position has a unique offset baked in
#   the model can distinguish "upper-left patch" from "lower-right"
class SingleHeadAttention(nn.Module):
    def __init__(self,d_models=768):
        super().__init__()

        self.Wq = nn.Linear(d_models,d_models,bias=False)
        self.Wk = nn.Linear(d_models, d_models, bias=False)
        self.Wv = nn.Linear(d_models, d_models, bias=False)

    def forward(self,x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        score = Q @ K.transpose(-2,-1)
        score = score / (d_models ** 0.5)

        weights = F.softmax(score)
        out = weights @ V
        return out , weights

attn = SingleHeadAttention()
x = torch.randn(2 , 197 ,768)
out,w = attn(x)

print(out.shape)      # → [2, 197, 768]   same shape as input ✓
print(w.shape)        # → [2, 197, 197]   attention map ✓
# print(w[0, 0].sum())  # → ~1.0             each row sums to 1 ✓

# ── inspect: where does CLS (token 0) attend most? ───────────
cls_attention = w[0, 0]   # row 0 = CLS token's attention weights
top5 = cls_attention.topk(5)
print("CLS attends most to positions:", top5.indices.tolist())
print("with weights:", [round(v, 3) for v in top5.values.tolist()])

class TransformerBlock(nn.Module):
    def __init__(self,d_model=768,n_heads=12,dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model) # prepares input for attention
        self.attn = nn.MultiheadAttention(
            d_model , n_heads ,
            dropout =  dropout,batch_first=True
        )
        #  If you used the same normalization:
        # 	•	The second sublayer (FFN) would get poorly scaled inputs
        # 	•	Training becomes unstable
        self.norm2 = nn.LayerNorm(d_model )# prepares input for FFN
        # Attention = mixes information between tokens
        # FFN = processes each token deeply
        self.ffn = nn.Sequential(
            nn.Linear(d_model , d_model * 4), # increasing “thinking space”
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4 , d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        normed =  self.norm1(x)
        out_attn , _ = self.attn(normed,normed,normed)
        x = x + out_attn # residual: ADD back to original x

        out_ffn = self.ffn(self.norm2(x))
        x = x + out_ffn

        return x

# ── shape check ───────────────────────────────────────────────
block = TransformerBlock()
x     = torch.randn(2, 197, 768)
out   = block(x)
print(out.shape)     # → [2, 197, 768]  ✓

# ── parameter breakdown ───────────────────────────────────────
total = sum(p.numel() for p in block.parameters())
print(f"One block params: {total/1e6:.2f}M")  # → ~7.09M

# Stage 5 — Stack 12 TransformerBlocks into the full encoder

depth = 12
# ── Option A: nn.Sequential ───────────────────────────────────
# Cleanest. One call runs all 12 blocks in order.
encoder_A = nn.Sequential(
    *[TransformerBlock() for _ in range(depth)]
)

x = torch.randn(2,197,768)
out , _ = encoder_A(x)
print(out.shape)
# ── Option B: nn.ModuleList ───────────────────────────────────
# Use this when you want to inspect each block's output.
encode_B = nn.ModuleList(
    [TransformerBlock() for _ in range(depth) ]
)

cls_norm = []

for i , block in enumerate(encode_B):
    x = block(x) #  runs one transformer block. Input is x, output overwrites x.
    # So after block 1, x is the updated sequence. After block 2,
    # x is updated again. Same variable, 12 rewrites.
    cls_vec = x[:,0,:] # cls at position 0
    cls_norm.append(cls_vec.norm().item())

print("\nCLS norm after each block:")
for i, norm in enumerate(cls_norm):
    bar = "█" * int(norm * 2)
    print(f"  Block {i+1:>2}  {norm:.3f}  {bar}")

# norm grows each block → CLS is accumulating information
# ── Parameter count ───────────────────────────────────────────
params = sum(p.numel() for p in encoder_A.parameters())
print(f"\nEncoder params: {params/1e6:.1f}M")   # → ~85.1M
# Each block ≈ 7.1M  ×  12 blocks  =  85.1M

class ViT(nn.Module):
    def __init__(
            self,
            img_size = 224 , patch_size = 16 , in_ch = 3,
            d_model = 768 , n_heads = 12 , depth = 12 ,
            dropout = 0.1 , num_classes = 1000
    ):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2 #196

        # Stage 1 — patch extraction + projection
        self.patch_emb = PatchEmb(img_size,patch_size,in_ch,d_model)

        # Stage 2 — CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1,self.n_patches+1,d_model))
        self.pos_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Stages 3+4 repeated × depth — the encoder
        self.block = nn.Sequential(
            *[TransformerBlock(d_model,n_heads,dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(d_model) # final normalization before output
        # input dim (B, d_model) Output → [B, 1000]
        self.head = nn.Linear(d_model , num_classes) # decision making choosing the correct class

    def forward(self,x):
        B = x.shape(0)

        # stage 1
        x = self.patch_emb(x)

        # stage 2
        cls = self.cls_token.expand(B,-1,-1) # [B,   1, 768]
        x = torch.cat([cls , x], dim = 1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_emb)

        # stage 3 + 4
        x = self.block(x)

        # stage 5
        x = self.norm(x)
        cls = x[:, 0] # only returning cls vector

        return self.head(cls)