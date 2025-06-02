"""
谣言检测系统 v3.0
全自动端到端版本
功能特点：
1. 单文本输入自动完成事件分类和谣言检测
2. 双模型协同架构
3. 生产级错误处理
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 系统配置
CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "num_events": 7,
    "event_embed_dim": 32,
    "hidden_dim": 128,
    "model_save_path": "./models",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class JointDataset(Dataset):
    """支持双任务训练的数据集"""

    def __init__(self, df, tokenizer):
        assert all(col in df.columns for col in ["text", "label", "event"])
        self.df = df[["text", "label", "event"]].dropna()
        self.tokenizer = tokenizer
        # 数据预处理
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.events = self.df["event"].astype(int).tolist()
        self._validate_data()

    def _validate_data(self):
        invalid_labels = [l for l in self.labels if l not in (0, 1)]
        invalid_events = [e for e in self.events if not 0 <= e <= 6]
        if invalid_labels or invalid_events:
            error_msg = []
            if invalid_labels:
                error_msg.append(f"无效label值数量: {len(invalid_labels)}")
            if invalid_events:
                error_msg.append(f"无效event值数量: {len(invalid_events)}")
            raise ValueError("数据校验失败 - " + ", ".join(error_msg))

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
            "event": torch.tensor(self.events[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class EventAwareRumorDetector(torch.nn.Module):
    """端到端谣言检测模型"""

    def __init__(self):
        super().__init__()
        # 共享BERT编码器
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # 事件分类模块
        self.event_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, CONFIG["num_events"])
        )
        # 谣言检测模块
        self.event_embed = torch.nn.Embedding(CONFIG["num_events"], CONFIG["event_embed_dim"])
        self.rumor_classifier = torch.nn.Sequential(
            torch.nn.Linear(768 + CONFIG["event_embed_dim"], CONFIG["hidden_dim"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(CONFIG["hidden_dim"], 2)
        )

    def forward(self, input_ids, attention_mask):
        # 共享特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]

        # 事件预测
        event_logits = self.event_classifier(pooled_output)

        # 谣言检测
        pred_events = torch.argmax(event_logits, dim=1)
        event_embeds = self.event_embed(pred_events)
        combined_feature = torch.cat([pooled_output, event_embeds], dim=1)
        rumor_logits = self.rumor_classifier(combined_feature)

        return event_logits, rumor_logits


def train_joint_model():
    """联合训练流程"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 数据加载
    train_df = pd.read_csv("./dataset/train.csv")
    val_df = pd.read_csv("./dataset/val.csv")
    train_dataset = JointDataset(train_df, tokenizer)
    val_dataset = JointDataset(val_df, tokenizer)
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    # 模型初始化
    model = EventAwareRumorDetector().to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # 训练循环
    best_f1 = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["attention_mask"].to(CONFIG["device"])
            }
            event_logits, rumor_logits = model(**inputs)
            # 双任务损失
            event_loss = torch.nn.functional.cross_entropy(
                event_logits, batch["event"].to(CONFIG["device"])
            )
            rumor_loss = torch.nn.functional.cross_entropy(
                rumor_logits, batch["label"].to(CONFIG["device"])
            )
            loss = 0.3 * event_loss + 0.7 * rumor_loss  # 可调权重
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        # 验证评估
        val_metrics = evaluate_joint_model(model, val_loader)
        print(f"\nEpoch {epoch + 1} 验证结果:")
        print(f"事件分类准确率: {val_metrics['event_acc']:.4f}")
        print(f"谣言检测F1: {val_metrics['rumor_f1']:.4f}")
        print(f"谣言检测准确率: {val_metrics['rumor_acc']:.4f}")
        # 保存最佳模型
        if val_metrics["rumor_f1"] > best_f1:
            best_f1 = val_metrics["rumor_f1"]
            save_path = CONFIG["model_save_path"]
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "joint_model.pt"))
            tokenizer.save_pretrained(save_path)


def evaluate_joint_model(model, loader):
    """联合评估"""
    model.eval()
    event_preds, event_true = [], []
    rumor_preds, rumor_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = {
                "input_ids": batch["input_ids"].to(CONFIG["device"]),
                "attention_mask": batch["attention_mask"].to(CONFIG["device"])
            }
            event_logits, rumor_logits = model(**inputs)
            # 事件分类
            event_preds.extend(torch.argmax(event_logits, dim=1).cpu().numpy())
            event_true.extend(batch["event"].numpy())
            # 谣言检测
            rumor_preds.extend(torch.argmax(rumor_logits, dim=1).cpu().numpy())
            rumor_true.extend(batch["label"].numpy())
    return {
        "event_acc": accuracy_score(event_true, event_preds),
        "rumor_f1": f1_score(rumor_true, rumor_preds, average="weighted"),
        "rumor_acc":accuracy_score(rumor_true,rumor_preds)
    }


class AutoDetector:
    """生产环境预测接口"""

    def __init__(self):
        self.device = CONFIG["device"]
        # 加载组件
        if not os.path.exists(CONFIG["model_save_path"]):
            raise RuntimeError("模型目录不存在，请先训练模型")
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_save_path"])
        # 初始化模型
        self.model = EventAwareRumorDetector()
        model_path = os.path.join(CONFIG["model_save_path"], "joint_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("模型文件不存在")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # 缓存初始化
        self.cache = {}

    def predict(self, text: str, cache_size: int = 100) -> dict:
        """带缓存的预测"""
        # 参数校验
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("输入文本必须是非空字符串")
        # 检查缓存
        text_hash = hash(text.strip())
        if text_hash in self.cache:
            return self.cache[text_hash]
        # 预处理
        encoding = self.tokenizer.encode_plus(
            text.strip(),
            max_length=CONFIG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        # 推理
        with torch.no_grad():
            event_logits, rumor_logits = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            event_probs = torch.softmax(event_logits, dim=1)
            rumor_probs = torch.softmax(rumor_logits, dim=1)
        # 结果处理
        result = {
            "event_type": torch.argmax(event_probs).item(),
            "event_confidence": event_probs.max().item(),
            "is_rumor": "Non-Rumor" if torch.argmax(rumor_probs).item() == 0 else "Rumor",
            "rumor_confidence": rumor_probs.max().item(),
            "detail": {
                "event_distribution": event_probs.squeeze().cpu().numpy().tolist(),
                "rumor_distribution": rumor_probs.squeeze().cpu().numpy().tolist()
            }
        }
        # 更新缓存
        if len(self.cache) >= cache_size:
            self.cache.popitem()
        self.cache[text_hash] = result
        return result


if __name__ == "__main__":
    # 训练模型
    print("启动联合训练...")
    train_joint_model()

    # 测试预测
    print("\n测试预测功能...")
    try:
        detector = AutoDetector()
        test_cases = [
            "Official announcement: All residents must evacuate immediately due to approaching hurricane",
            "Breaking: 5G networks confirmed to spread coronavirus",
            "Recent study shows chocolate improves cognitive function",
            "BREAKING: New study finds face masks cause oxygen deprivation",
            "Feni Trunk Road is crowded with people for AL rally",
            "Khaleda again in High Court for bail in Comilla murder case",
            "Truck falls into river after bridge collapses in Mahalchhari, 1 missing",
            "BREAKING: Gunman alleged to have taken hostages in #sydneysiege identified as Man Haron Monis, official tells CNN http://t.co/VUVPPrKuDc"
        ]
        for text in test_cases:
            result = detector.predict(text)
            print(f"\n输入文本: {text[:60]}...")
            print(f"预测事件类型: {result['event_type']} (置信度: {result['event_confidence']:.2%})")
            print(f"谣言判定: {result['is_rumor']} (置信度: {result['rumor_confidence']:.2%})")
    except Exception as e:
        print(f"预测失败: {str(e)}")