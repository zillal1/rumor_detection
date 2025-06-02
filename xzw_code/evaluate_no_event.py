"""
适配纯文本模型的评估脚本
修改点：移除事件相关处理，适配基础文本分类器
"""
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from model import TextClassifier
# 配置需与文本模型训练时一致
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "model_save_path": "./text_rumor_detector",  # 确保路径正确
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class TextTestDataset(Dataset):
    """适配纯文本模型的数据集"""
    def __init__(self, df, tokenizer):
        # 只保留必要字段
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer

        # 验证标签范围
        invalid_labels = [l for l in self.labels if l not in (0, 1)]
        if invalid_labels:
            raise ValueError(f"发现{len(invalid_labels)}个无效标签（必须为0或1）")

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
            "true_label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextModelEvaluator:
    def __init__(self):
        self.device = CONFIG["device"]
        # 加载组件
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_save_path"])
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """加载纯文本分类模型结构"""
        model = TextClassifier()  # 使用原始训练代码中的模型类
        model_path = os.path.join(CONFIG["model_save_path"], "model.bin")  # 注意文件名
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def evaluate(self, test_csv_path):
        """执行评估"""
        # 加载数据
        test_df = pd.read_csv(test_csv_path)
        print(f"\n数据集样本数量: {len(test_df)}")
        print("前3条样例:")
        print(test_df[["id", "text"]].head(3))

        # 创建数据集
        dataset = TextTestDataset(test_df, self.tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"])

        # 执行预测
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="预测进度"):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()

                true_labels.extend(batch["true_label"].numpy())
                pred_labels.extend(batch_preds)

        # 直接使用原始标签映射（0:非谣言，1:谣言）
        print("\n评估结果:")
        print(f"准确率: {accuracy_score(true_labels, pred_labels):.4f}")
        print(f"F1分数: {f1_score(true_labels, pred_labels):.4f}")
        print("\n混淆矩阵:")
        print(confusion_matrix(true_labels, pred_labels))
        print("\n分类报告:")
        print(classification_report(true_labels, pred_labels,
                                   target_names=["Non-Rumor", "Rumor"]))

        # 保存结果
        test_df["pred_label"] = pred_labels
        save_path = os.path.join(CONFIG["model_save_path"], "text_test_predictions.csv")
        test_df.to_csv(save_path, index=False)
        print(f"\n预测结果已保存至: {save_path}")

if __name__ == "__main__":
    evaluator = TextModelEvaluator()
    test_csv_path = "./dataset/val.csv"  # 修改为实际路径

    try:
        evaluator.evaluate(test_csv_path)
    except Exception as e:
        print(f"评估失败: {str(e)}")