import joblib
import re

class RumourDetectClass:
    def __init__(self):
        # 加载模型和vectorizer
        model_data = joblib.load('lr_model.pkl')
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']

    def preprocess(self, text):
        # 与训练时一致的小写和去标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def classify(self, text: str) -> int:
        """
        对输入的文本进行谣言检测
        Args:
            text: 输入的文本字符串
        Returns:
            int: 预测的类别（0表示非谣言，1表示谣言）
        """
        text = self.preprocess(text)
        X_vec = self.vectorizer.transform([text])
        pred = self.model.predict(X_vec)[0]
        return int(pred)