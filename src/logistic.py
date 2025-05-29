import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 读取数据
train_df = pd.read_csv('../dataset/split/train.csv')
val_df = pd.read_csv('../dataset/split/val.csv')

# 特征和标签
X_train = train_df['text']
y_train = train_df['label']
X_val = val_df['text']
y_val = val_df['label']

# 文本预处理
X_train = X_train.str.lower().str.replace('[^\w\s]', '', regex=True)
X_val = X_val.str.lower().str.replace('[^\w\s]', '', regex=True)

# 文本向量化（TF-IDF）
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 验证集评估
val_pred = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_pred)
print(f'Val Acc: {val_acc:.4f}')
print(classification_report(y_val, val_pred))

# 保存模型和向量器
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'lr_model.pkl')
print('模型已保存为lr_model.pkl')