# https://chatgpt.com/share/69ef33b2-d528-83e8-a049-ccd89e4aebb0
# https://claude.ai/share/f5f6efec-9f4e-463b-9b35-a10cd47d2e31



from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup


def get_dataloader(batch_size = 32 , max_length = 128):
    # ── 1. Load raw dataset ──────────────────────────────────
    # Downloads SST-2 from HuggingFace Hub (cached after first run)
    # dataset["train"]      → 67,349 examples
    # dataset["validation"] → 872 examples
    # each example: {"sentence": "...", "label": 0 or 1, "idx": N}
    dataset = load_dataset("glue","sst2")

    # ── 2. Load BERT tokenizer ───────────────────────────────
    # This is WordPiece, vocab=30,522
    # "uncased" means it lowercases everything before tokenizing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # can go with only bert also

    def tokenize(batch):
        return tokenizer(batch["sentence"],padding="max_length",truncation=True,max_length=max_length,return_tensors=False)

    tokenized = dataset.map(tokenize,batched=True)# batched=True means it processes 1000 examples at a time (faster)

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )
    # input_ids      → token IDs, shape (max_length,)
    # attention_mask → 1 for real tokens, 0 for padding
    # token_type_ids → all 0s for single-sentence tasks (used in NSP)
    # label          → 0 (negative) or 1 (positive)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader,test_loader

class bertClassifier(nn.Module):
    def __init__(self,num_label=2,dropout=0.1):
        super().__init__()
        # self.bert is the model which we are fine tuning does not learn any thing new
        # classifier is new and learns from scratch, while BERT is pretrained and fine tuned.
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.dropout = nn.Dropout(dropout) # using dropout for prevent overfitting / regularization

        #[CLS] token contain sentence summary.
        self.classifier = nn.Linear(768,num_label) # self.classifier use cls token vector and give output positive negative

    def forward(self,input_ids , attention_mask , token_type_ids):
        # we are give our data set to this pre-trained model for fine tuning
        # our data have padding and all therefore we are providing attention mask and input ids
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

        #   outputs.pooler_output     → shape: (batch, 768)
        #   pooler_output is the CLS token passed through a Linear+Tanh
        #   (BERT's own pooling layer, trained during pretraining)
        cls_output = outputs.pooler_output # use BERT's built-in pooler

        # cls_output = outputs.last_hidden_state[:, 0, :] grab [CLS] manually

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

def evalute(model , test_loader , device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids , attention_mask , token_type_ids)

            prediction = torch.argmax(logits, dim = 1)

            correct += (prediction == labels ).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total

def train():

    epochs = 3
    batch_size = 32
    max_length = 128
    lr = 2e-5
    warmup_frac = 0.1 # 10% of steps use for warmup

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # data
    train_loader , test_loader = get_dataloader(batch_size , max_length)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(test_loader)}")

    # model
    model = bertClassifier(num_label=2).to(device)

    # optimizer
    # AdamW with weight decay on weights, NOT on biases/LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_group_parameter = [
        {
            # n -> name of parameter and p-> actual tensor  nd -> no_decay
            "params": [p for n , p in model.named_parameters() if not any (nd in n for nd in no_decay)],
            "weight_decay":0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.00,
        },
    ]
    optimizer = optim.AdamW(optimizer_group_parameter,lr=lr)

    # lr schedular
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_frac)

    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps
    )
    # LR rises linearly for warmup_steps, then decays linearly to 0

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step , batch in enumerate(train_loader):
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            token_type_ids  = batch["token_type_ids"].to(device)
            labels          = batch["label"].to(device)

            logits = model(input_ids , attention_mask , token_type_ids)
            loss = loss_function(logits , labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            schedular.step()

            total_loss += loss.item()

            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch + 1} | Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

            # ── Validation after each epoch ───────────────────────
        val_acc = evalute(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} complete | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}\n")

        # ── Save the model ────────────────────────────────────────
    torch.save(model.state_dict(), "bert_sst2.pt")
    print("Model saved to bert_sst2.pt")


if __name__ == "__main__":
    train()

