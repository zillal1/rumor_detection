"""
新测试集评估脚本
功能：使用训练好的模型评估新数据集性能
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
from ourmodel_new import EventAwareRumorDetector
# 配置需与训练时一致
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "model_save_path": "./models",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class TestDataset(Dataset):
    """测试数据集类"""

    def __init__(self, df, tokenizer):
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


class ModelEvaluator:
    def __init__(self):
        self.device = CONFIG["device"]
        # 加载组件
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_save_path"])
        # 加载模型结构（根据实际模型类调整）
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """加载模型结构（需与训练代码一致）"""
        # 此处使用之前定义的EventAwareRumorDetector，如结构有变需修改
        model = EventAwareRumorDetector()
        model_path = os.path.join(CONFIG["model_save_path"], "joint_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def evaluate(self, test_csv_path):
        """执行完整评估流程"""
        # 加载数据
        test_df = pd.read_csv(test_csv_path)
        print(f"\n数据集样本数量: {len(test_df)}")
        print("前3条样例:")
        print(test_df[["id", "text"]].head(3))

        # 创建数据集
        dataset = TestDataset(test_df, self.tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"])

        # 执行预测
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="预测进度"):
                # 数据转移到设备
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                # 模型推理
                _, rumor_logits = self.model(**inputs)
                batch_preds = torch.argmax(rumor_logits, dim=1).cpu().numpy()

                # 收集结果
                true_labels.extend(batch["true_label"].numpy())
                pred_labels.extend(batch_preds)

        # 标签映射转换
        """
        解释映射关系：
        - 模型输出0表示"Rumor"（应映射到测试集的1）
        - 模型输出1表示"Non-Rumor"（应映射到测试集的0）
        """
        adjusted_preds = pred_labels
        adjusted_true = np.array(true_labels)

        # 计算指标
        print("\n评估结果:")
        print(f"准确率: {accuracy_score(adjusted_true, adjusted_preds):.4f}")
        print(f"F1分数: {f1_score(adjusted_true, adjusted_preds):.4f}")
        print("\n混淆矩阵:")
        print(confusion_matrix(adjusted_true, adjusted_preds))
        print("\n分类报告:")
        print(classification_report(adjusted_true, adjusted_preds, target_names=["Non-Rumor", "Rumor"],labels=[0,1]))

        # 保存预测结果
        test_df["pred_label"] = adjusted_preds
        save_path = os.path.join(CONFIG["model_save_path"], "test_predictions.csv")
        test_df.to_csv(save_path, index=False)
        print(f"\n预测结果已保存至: {save_path}")



if __name__ == "__main__":
    # 初始化评估器
    evaluator = ModelEvaluator()

    # 执行评估
    test_csv_path = "./dataset/train2.csv"  # 修改为实际路径
    try:
        evaluator.evaluate(test_csv_path)
    except Exception as e:
        print(f"评估失败: {str(e)}")