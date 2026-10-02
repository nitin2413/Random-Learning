import torch
import torch.nn as nn
import torch.optim as optim


text = "hello world hello world hello world"
chars = list(set(text))
char2idx = {ch : i for i , ch in enumerate(chars)}
idx2char = {i : ch for i , ch in enumerate(chars)}

data = [char2idx[ch] for ch in text]

X = torch.tensor(data[:-1])
Y = torch.tensor(data[1:])

input_size = len(chars)

def onehot(x,size):
    vec = torch.zeros(size)
    vec[x] = 1
    return vec

X_onehot = torch.stack([onehot(x,input_size) for x in X])

X_onehot = X_onehot.unsqueeze(1)
Y = Y.unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=False)
        self.lin = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden):
        out, hidden = self.lstm(x,hidden)
        out = self.lin(out)
        return out , hidden

class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.gru = nn.GRU(input_size,hidden_size,batch_first=False)
        self.lin = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden):
        out, hidden = self.gru(x,hidden)
        out = self.lin(out)
        return out, hidden

hidden_size = 8
model1 = LSTMModel(input_size,hidden_size,input_size)
model2 = GRUModel(input_size,hidden_size,input_size)
loss_fun_ = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

# epochs = 100
# for epoch in range(epochs):
#     hidden_gate = torch.zeros(1,1,hidden_size)
#     cell_gate = torch.zeros((1,1,hidden_size))
#     hidden = (hidden_gate,cell_gate)
#
#     output , hidden = model(X_onehot, hidden)
#     loss = loss_fun_(output.view(-1,input_size) , Y.view(-1))
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         _, pred = torch.max(output, 2)
#         pred_text = ''.join([idx2char[i.item()] for i in pred.squeeze()])
#         print(f"[LSTM] Epoch {epoch + 1}, Loss: {loss.item():.4f}, Prediction: {pred_text}")


epochs = 100
for epoch in range(epochs):
    # --- Train LSTM ---
    hidden_lstm = (torch.zeros(1,1,hidden_size),
                   torch.zeros(1,1,hidden_size))

    out_lstm, hidden_lstm = model1(X_onehot, hidden_lstm)
    loss1 = loss_fun_(out_lstm.view(-1,input_size), Y.view(-1))

    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    # --- Train GRU ---
    hidden_gru = torch.zeros(1,1,hidden_size)

    out_gru, hidden_gru = model2(X_onehot, hidden_gru)
    loss2 = loss_fun_(out_gru.view(-1,input_size), Y.view(-1))

    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    # --- Print progress ---
    if (epoch + 1) % 10 == 0:
        _, pred_lstm = torch.max(out_lstm, 2)
        pred_text_lstm = ''.join([idx2char[i.item()] for i in pred_lstm.squeeze()])

        _, pred_gru = torch.max(out_gru, 2)
        pred_text_gru = ''.join([idx2char[i.item()] for i in pred_gru.squeeze()])

        print(f"[LSTM] Epoch {epoch+1}, Loss: {loss1.item():.4f}, Prediction: {pred_text_lstm}")
        print(f"[GRU ] Epoch {epoch+1}, Loss: {loss2.item():.4f}, Prediction: {pred_text_gru}")