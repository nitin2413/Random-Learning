import torch
import  torch.nn  as  nn
import torch.optim as optim

text = "hello world hello world hello world"
chars = list(set(text))
char2idx = {ch : i for i, ch in enumerate(chars)}#char2idx → maps character → index
idx2char = {i : ch for i, ch in enumerate(chars)}#idx2char → maps index → character

data = [char2idx[ch] for ch in text]

X = torch.tensor(data[:-1]) # all data except last
Y = torch.tensor(data[1:]) # all data except first (as we can't predict first letter)

input_size = len(chars)
def one_hot(x,size):
    vec = torch.zeros(size)
    vec[x] = 1 # if x = 0 [0,0,0,0] at zero position add 1 [1,0,0,0]
    return vec

X_onehot = torch.stack([one_hot(x,input_size) for x in X]) # storing all returned value of onehot in X_onehot list.

#An RNN in PyTorch expects input in 3D tensor form
# currently X_onehot is 2D , we use unsqueeze to make it 3D without tampering the original data
# Reshape for RNN: (seq_len, batch, input_size)

X_onehot = X_onehot.unsqueeze(1)
Y = Y.unsqueeze(1)

class RnnModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        # batch_first just specifies the dimension in which we give input
        self.rnn = nn.RNN(input_size ,hidden_size,batch_first= False)
        self.lin = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden):
        out,hidden = self.rnn(x,hidden)
        out = self.lin(out)#out in PyTorch ≈ list of self.last_hs[t] in NumPy.
        return out,hidden#hidden in PyTorch ≈ the last entry of self.last_hs (self.last_hs[len(inputs)-1])

hidden_size = 8
# input_size = number of unique text over here 4
# output_size = how many unique characters we can predict.
# Since both sets are the same vocabulary, they must be equal.
model = RnnModel(input_size , hidden_size, input_size)

loss_fun_= nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 100
for epoch in range(epochs):
    hidden = torch.zeros(1,1,hidden_size)# ensuring same dimension
    output , hidden = model(X_onehot , hidden)

    loss = loss_fun_(output.view(-1,input_size), Y.view(-1))
    # over here we squeeze 3D predicted to 2D and 2D ture label to 1D
    # it works because PyTorch’s loss function is designed for this exact shape:
    # 2D predictions vs. 1D class labels.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        #finds the index of the maximum predicted
        _, pred = torch.max(output, 2)
        #squeeze remove extra data .item convert tensor value to int join concat predicted char
        pred_text = ''.join([idx2char[i.item()] for i in pred.squeeze()])
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Prediction: {pred_text}")


