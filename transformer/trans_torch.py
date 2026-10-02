import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Scaledotproduct(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K , V , mask = None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q , K.transpose(-2,-1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # use in decoder part to hide future data from input

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights,nan=0.0)
        weights = self.dropout(weights)

        return torch.matmul(weights , V) , weights

attn = Scaledotproduct()
batch , seq_len , d_k = 1 , 5, 32
Q = torch.randn(batch , seq_len , d_k)
K = torch.randn(batch , seq_len , d_k)
V = torch.randn(batch , seq_len , d_k)

our_out , our_w = attn(Q , K , V)
# in build torch attention calculation
torch_out = F.scaled_dot_product_attention(Q,K,V)
print("MAX Difference:",(our_out - torch_out).abs().max().item())# .abs remove the sign (+,-)

tokens = ["<s>", "The", "cat", "sat", "on", "mat"]
seq = len(tokens)
d_k = 64

Q = torch.randn(batch, seq , d_k)
K = torch.randn(batch, seq , d_k)
V = torch.randn(batch, seq , d_k)

_ , weights = attn(Q, K , V)
w = weights[0].detach().numpy()  # (seq, seq)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(w, cmap='viridis', vmin=0, vmax=w.max())
ax.set_xticks(range(seq)); ax.set_xticklabels(tokens)
ax.set_yticks(range(seq)); ax.set_yticklabels(tokens)
ax.set_title("Attention weights — each row attends to columns")
plt.colorbar(im)

# annotate each cell
for i in range(seq):
    for j in range(seq):
        ax.text(j, i, f"{w[i,j]:.2f}", ha='center', va='center',
                color='white' if w[i,j] < w.max()*0.6 else 'black', fontsize=7)

plt.tight_layout()
plt.savefig("attention_weights.png", dpi=150)
plt.show()
