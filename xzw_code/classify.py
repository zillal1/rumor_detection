import os
import torch
from transformers import BertTokenizer, BertModel

CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "model_save_path": "./text_rumor_detector",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class TextPredictor:
    """文本预测接口"""

    def __init__(self):
        self.device = CONFIG["device"]

        # 加载组件
        if not os.path.exists(CONFIG["model_save_path"]):
            raise FileNotFoundError(f"模型目录不存在: {CONFIG['model_save_path']}")

        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_save_path"])

        # 初始化模型
        self.model = TextClassifier()
        model_path = os.path.join(CONFIG["model_save_path"], "model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def classify(self, text: str) -> int:
        # 参数校验
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("text参数必须为非空字符串")

        # 预处理
        encoding = self.tokenizer.encode_plus(
            text.strip(),
            max_length=CONFIG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 预测
        with torch.no_grad():
            inputs = {
                "input_ids": encoding["input_ids"].to(self.device),
                "attention_mask": encoding["attention_mask"].to(self.device)
            }
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        # 这里用于详细信息的展示，可以去除注释
        # result = {
        #     "prediction": "Non-Rumor" if torch.argmax(probs).item() == 0 else "Rumor",
        #     "confidence": probs.max().item(),
        #     "class_probabilities": {
        #         "Rumor": probs[0][0].item(),
        #         "Non-Rumor": probs[0][1].item()
        #     }
        # }
        # print("\n预测结果:")
        # print(f"文本: {text}")
        # print(f"预测结论: {result['prediction']}")
        # print(f"置信度: {result['confidence']:.2%}")
        # print(f"详细概率: {result['class_probabilities']}")

        return torch.argmax(probs).item()


if __name__ == "__main__":
    # 训练流程

    # 预测示例
    print("\n测试预测功能...")
    predictor = TextPredictor()
    test_case = "Breaking: 5G networks confirmed to spread coronavirus"
    result = predictor.classify(test_case)
    print(result)

