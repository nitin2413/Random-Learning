from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re


# ==== Load vocab ====
with open("vocab.pkl", "rb") as f:
    itos, stoi, PAD, UNK = pickle.load(f)

# ==== Tokenizer & numericalize ====
def tokenize(text: str):
    # keep only words/numbers and apostrophes, lowercased
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def numericalize(text: str):
    tokens = tokenize(text)
    if len(tokens) == 0:
        # avoid empty sequence
        return [UNK]
    return [stoi.get(tok, UNK) for tok in tokens]

# ==== Model ====
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden=256, num_layer=1,
                 output_class=2, pad_idx=PAD, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layer,
            batch_first=True,
            bidirectional=True
        )
        self.lin = nn.Linear(hidden * 2, output_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # x: [batch, seq_len]
        em = self.emb(x)  # [batch, seq_len, embed_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: [num_layers * num_directions, batch, hidden]
        # last layer forward = -2, backward = -1
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, hidden*2]
        out = self.dropout(h_cat)
        return self.lin(out)  # [batch, output_class]

# ==== Device & load model ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# for server you can also force cpu: device = torch.device("cpu")

model = BiLSTMClassifier(vocab_size=len(itos)).to(device)
state = torch.load("imdb.pth", map_location=device)
model.load_state_dict(state)
model.eval()

# ==== API ====
app = FastAPI()

class Review(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "IMDB sentiment model API running"}

@app.post("/predict")
def predict_sentiment(review: Review):
    tokens = numericalize(review.text)
    lens = torch.tensor([len(tokens)], dtype=torch.long)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    x, lens = x.to(device), lens.to(device)

    with torch.no_grad():
        logits = model(x, lens)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    sentiment = "Positive" if pred == 1 else "Negative"
    return {
        "sentiment": sentiment,
        "class_index": pred,
        "probabilities": probs
    }
