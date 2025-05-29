import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re

# 超参数设置
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
MAX_LEN = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 简单分词器
def tokenize(text):
    return re.findall(r'\w+', text.lower())

# 构建词表
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

class RumorDataset(Dataset):
    # 谣言数据集，返回文本和标签
    def __init__(self, df, vocab):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

class BiGRU(nn.Module):
    # BiGRU模型定义
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # 前向传播
        emb = self.embedding(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        out = self.fc(h)
        return out.squeeze(1)

def evaluate(model, loader):
    # 评估函数，返回准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    # 读取数据集
    train_df = pd.read_csv('../dataset/split/train.csv')
    val_df = pd.read_csv('../dataset/split/val.csv')

    # 构建词表
    vocab = build_vocab(train_df['text'])
    # 构建数据集
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    # 初始化模型、优化器和损失函数
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}, Val Acc: {val_acc:.4f}')
        
    # 保存模型checkpoint
    torch.save(model.state_dict(), 'bigru.pt')
    print('模型已保存为bigru.pt')

if __name__ == '__main__':
    main() 