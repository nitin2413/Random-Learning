import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1 , keepdims = True))
    return e / np.sum(e,axis=-1,keepdims=True)

def attention (Q , K , V):
    d_k = Q.shape[-1] # take value of last dim of Q (q(4,8) value of d_k = 8)
    sim_score = np.matmul(Q , K.transpose(0, 2, 1))  # swap last two dims # each token attends to every token
    scaling = np.sqrt(d_k) # Dot products become very large when dimension increases therefor this is sequrity measure
    score = sim_score / scaling
    weights = softmax(score) # attention probabilities
    output = np.matmul(weights , V) # (4,4) × (4,8) → (4,8) , Each token receives a weighted combination of values
    return  output , weights

np.random.seed(42)
batch , seq_len , d_k = 2 , 4 , 8
Q = np.random.randn(batch ,seq_len , d_k)
K = np.random.randn(batch ,seq_len , d_k)
V = np.random.randn(batch ,seq_len , d_k)

out , weight = attention(Q , K , V)

print("Scores shape:", (Q @ K.transpose(0,2,1)).shape)   # (4, 4)
print("Weights shape:", weight.shape)           # (4, 4)
print("Output shape:", out.shape)          # (4, 3)
print("Row sums (must be 1.0):", weight.sum(axis=-1))  # [1. 1. 1. 1.]
print("\nAttention weights:\n", np.round(weight, 3))