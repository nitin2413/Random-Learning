# har har mahadev #
# ====== IMDB SENTIMENT CLASSIFICATION: Run using gpu ======

# ======imports=====

import re  # regular expression (regex) for simple tokenization
from collections import Counter  # counter to count duplicates of token
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from pothole import load_dataset


# =======SEED======

Seed = 42
random.seed(Seed)
torch.manual_seed(Seed)

# =======LOAD DATA SETS IMDB========

imdb = load_dataset("imdb") # contain 25k each test and train dataset

def sub_samples( ds, n ): # sub_sampling so that we use less data
    # take n number of data_set from imdb example 10k
    return ds.select(range(min(n,len(ds))))

train_raw = sub_samples(imdb["train"],10000)
test_raw = sub_samples(imdb["test"],5000)

# ======= tokenize ============

def tokenize(text : str):
    # text : str parameter entered should be string
    # token are made by categorical eg [2003,nitin,can't....]
    return re.findall(r"[A-Za-z0-9']+",text.lower())

# ======== BUILD VOCAB ==========

counter = Counter()
for ex in train_raw:
    counter.update(tokenize(ex["text"]))# map count of each repeated token in train_raw

special = ["<pad>","<unk>"] # special token for padding and unknown text
Vocab_size = 25000 # all 25k words are unique
# counter.most_common tell how many time word is repeated
# loop (w) is used to keep only word and (_) for count
most_common = [w for w,_ in counter.most_common(Vocab_size-len(special))]
itos = special + most_common # index -> token
stoi = {w:i for i ,w in enumerate(itos)} # mapping token -> index
PAD , UNK = stoi["<pad>"] , stoi["<unk>"]

def numericalize(text):
    # convert text -> token ids (unknown text -> unk)
    # stoi.get check weather tok exist in vocabulary then add to list if not change it with unk and add to list
    return [stoi.get(tok,UNK) for tok in tokenize(text)]

#=========== Datasets(bridge IMDB dataset → PyTorch.) ============
# pytorch handel tensor data not numpy

class IMDBDatasets(Dataset):
    def __init__(self,hf_split):#hf_split is a Hugging Face split (like train_raw or test_raw).
        # see numericalize comment , item map token ids with their label in a single list
        # example ([12, 54, 789, 3, 88], 0),
        self.items =  [(numericalize(ex["text"]),ex["label"]) for ex in hf_split]
        # pytorch expect len and getitem function
    def __len__(self):# len gives total number of pairs of text and their label
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]# return item based on their given index 1 = 5 gives 6th (token id,label)

train_ds = IMDBDatasets(train_raw)
test_ds = IMDBDatasets(test_raw )

#============== collate(pad per batch) ================
# add padding to every token id till the range of longest token id as we can stack pothole of same size in torch

def collate(batch , pad_idx= PAD):
    seqs , ys = zip(*batch) # unpack the train data sets text and labels
    # compute original lengths of seq before padding as we need it later in lstm model to skip padded token id
    lens = torch.tensor([len(s) for s in seqs],dtype=torch.long)
    maxlen = int(lens.max())
    # torch.full create a tensor of given shape , where every element is filled with same constant value
    # len(seqs) give size of a seqs so that we can match it with maxlen size with
    x = torch.full((len(seqs),maxlen),pad_idx,dtype=torch.long)
    for i , s in enumerate(seqs):
        # x become our main input data as it gets organise in tensor format
        # i is the index of the sequence in the batch.
        # :len(s) → slice the first part of row i, up to the true sequence length.
        x[i , :len(s)] = torch.tensor(s,dtype=torch.long)
        # seqs = [[5,7,8], [3,9]]
        # x =[[5,7,8,0,0],[3,9,0,0,0]]
    y = torch.tensor(ys,dtype=torch.long)
    return x,y,lens

#dataloader pass all required argument to collate
train_loader = DataLoader(train_ds ,batch_size=64,shuffle=True,collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=64,shuffle=False,collate_fn=collate)

#============ BiLSTMClassifier ===============

class BiLSTMClassifier(nn.Module):
    # embed_dim = 200 give vector of size 200 , in one_hot suppose 1000 unique dataset then [0,0,1,0,1,0....1000]
    # onehot is bad for large vocab_size where as vector are fix
    # example dog and cat have closer vector than dog and computer
    def __init__(self,vocab_size,embed_dim=200,hidden=256,num_layer=1,output_class=2,pad_idx=PAD,dropout=0.3):
        super().__init__()
        # emb shape shape from B,T TO B,T,E
        self.emb = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        # batch_first just specifies the dimension in which we give input
        self.lstm = nn.LSTM(embed_dim,hidden,num_layer,batch_first=True,bidirectional=True)
        #hidden into 2 because model being bidirectional
        self.lin = nn.Linear(hidden*2,output_class)
        self.dropout = nn.Dropout(dropout)

    def forward (self,x,lengths):
        # em add the dimension x : (B,T)
        # Converts token IDs → dense vectors
        em = self.emb(x) # (B,T,E)
        # pack_padded_sequence compresses the padded batch together with lengths information so the LSTM ignores padded
        # Pass lengths.cpu() because pack_padded_sequence expects CPU lengths in some versions.
        # enforce_sorted=False → we don’t need to sort sequences by length manually.
        packed = nn.utils.rnn.pack_padded_sequence(em,lengths.cpu() ,batch_first=True,enforce_sorted=False)
        #packed_out -> “all the intermediate LSTM outputs, but stored in a smart compressed way
        # we are not using it anywhere , what we are using is final hidden state
        packed_out , (h_n,c_n) = self.lstm(packed)
        final_fow_h = h_n[-2] # → forward LSTM’s last hidden state
        final_bak_h = h_n[-1] # → backward LSTM’s last hidden state
        h_cat = torch.cat([final_fow_h,final_bak_h],dim=1) # (B,hidden*2)
        out = self.dropout(h_cat)
        logits = self.lin(out)
        return logits

def accuracy(logits,y):
    # chooses the predicted class index per example. == y yields boolean tensor where predictions are correct.
    #.item convert scalar tensor to python float
    return (logits.argmax(dim=1) == y).float().mean().item()

# evaluation record loss matrics and do not update
def evaluate(model,loader,loss_function,device):
    model.eval()
    total_loss = total_acc = n = 0
    with torch.no_grad():
        for xb , yb , lb in loader:
            xb , yb , lb = xb.to(device), yb.to(device), lb.to(device)
            logits = model(xb,lb)
            loss = loss_function(logits,yb)
            m = xb.size(0)
            total_loss += loss.item()*m
            total_acc += accuracy(logits,yb)*m
            n += m
    return total_loss / n, total_acc / n

device = torch.device("mps" if torch.mps.is_available() else "cpu")
model = BiLSTMClassifier(vocab_size=len(itos)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=2e-3)

epochs = 5
best_val = float("inf") # initialized for tracking best validation loss if you want to save the best model

for epoch in range(epochs):
    model.train()
    tr_loss = tr_acc = n = 0
    for xb , yb , lb , in train_loader :
        xb , yb , lb = xb.to(device), yb.to(device) , lb.to(device)
        # lb is lengths (true sequence lengths before padding)
        # lb is not true label as they are already concatenated during hf_split
        logits = model(xb,lb) # predicted value
        loss = loss_function(logits,yb)
        optimizer.zero_grad()
        loss.backward()
        # prevent exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        m =xb.size(0) #batch size (number of samples in this batch)
        tr_loss += loss.item()*m
        tr_acc += accuracy(logits,yb)*m
        n += m #  total number of samples seen so far.

    val_loss , val_acc = evaluate(model, test_loader ,loss_function ,device)
    print(f"epoch {epoch:02d} | train loss {tr_loss/n:.04f} | acc {tr_acc:.03f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")


torch.save(model.state_dict(),"imdb.pth")
