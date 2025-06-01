import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 系统配置
CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "model_save_path": "./text_rumor_detector",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class TextDataset(Dataset):
    """文本数据集加载器"""

    def __init__(self, df, tokenizer):
        # 数据校验
        assert all(col in df.columns for col in ["text", "label"]), "缺少必要数据列"

        # 数据预处理
        self.df = df[["text", "label"]].dropna()
        self.tokenizer = tokenizer

        # 转换数据类型
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

        # 验证数据范围
        self._validate_data()

    def _validate_data(self):
        """验证数据有效性"""
        invalid_labels = [l for l in self.labels if l not in (0, 1)]

        if invalid_labels:
            raise ValueError(f"发现{len(invalid_labels)}个无效label值（需为0或1）")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=CONFIG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TextClassifier(torch.nn.Module):
    """纯文本分类模型"""

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.classifier(outputs.last_hidden_state[:, 0, :])


def evaluate(model, loader):
    """模型评估"""
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = {
                "input_ids": batch["input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["attention_mask"].to(CONFIG["device"])
            }
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(batch["label"].numpy())

    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds, average="weighted")
    }


def train():
    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 加载数据
    print("\n" + "=" * 50)
    print("加载数据...")
    train_df = pd.read_csv("./dataset/train.csv")
    val_df = pd.read_csv("./dataset/val.csv")

    # 创建数据集
    print("创建数据集...")
    train_dataset = TextDataset(train_df, tokenizer)
    val_dataset = TextDataset(val_df, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"]
    )

    # 初始化模型
    print("初始化模型...")
    model = TextClassifier().to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    print("开始训练...")
    best_f1 = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            inputs = {
                "input_ids": batch["input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["attention_mask"].to(CONFIG["device"])
            }

            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs, batch["label"].to(CONFIG["device"])
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": loss.item()})

        # 验证评估
        avg_loss = epoch_loss / len(train_loader)
        metrics = evaluate(model, val_loader)
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"平均训练损失: {avg_loss:.4f}")
        print(f"验证集准确率: {metrics['accuracy']:.4f}")
        print(f"验证集F1分数: {metrics['f1']:.4f}")

        # 保存最佳模型
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_path = CONFIG["model_save_path"]
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.bin"))
            tokenizer.save_pretrained(save_path)
            print(f"模型已保存至 {save_path}")


if __name__ == "__main__":
    # 训练流程
    train()
