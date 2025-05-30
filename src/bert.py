import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel

# 参数设置
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集类
class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_ids, attention_mask, label

# 模型类
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_output)
        return logits.squeeze(1)

# 评估函数
def evaluate_bert(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 主函数
def main():
    train_df = pd.read_csv("../dataset/split/train.csv")
    val_df = pd.read_csv("../dataset/split/val.csv")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_set = BertDataset(train_df, tokenizer, max_len=MAX_LEN)
    val_set = BertDataset(val_df, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = BertClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_bert(model, val_loader)
        print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "bert_rumor.pt")
    print("模型已保存为 bert_rumor.pt")

if __name__ == "__main__":
    main()
