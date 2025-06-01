## 谣言检测模型接口`classify.py`使用说明

##### 写在前面：

使用版本说明：

```
python>3.0.0
pytorch<2.0.0 (如果需要)
```

本次实验中的模型使用了`torch`为`2.0.0`版本的，是因为下载更早的版本较为困难，但是所用代码没有超出`2.0.0`版本之前范围。

---

从交大云盘上下载完整目录

如果是单独下载模型，需要保证模型在对应的文件目录下

在`classify.py`中有对应的文件路径，为`./text_rumor_detector`：

```python
CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "model_save_path": "./text_rumor_detector",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
```

在python文件中`import classify`之后可以按照以下的代码进行调用

```python
predictor = classify.TextPredictor() # 先进行模型初始化
………………………………………
result=predictor.classify(text)      # text为一个str类型的变量，result为0/1
```

调用函数的一个例子（classify和该py文件为于同一目录下）

```python
import classify
print("\n测试预测功能...")
predictor = classify.TextPredictor()
test_case =  "Breaking: 5G networks confirmed to spread coronavirus"
result = predictor.classify(test_case)
print(result)
```

---

> [!CAUTION]

这里classify函数的返回值为0/1，感觉只输出0/1过于单调了，所以在classify中有一段注释掉的部分，可以输出判断内容的置信度等

```python
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
```

去除注释之后对于单个谣言检测的输出会更加立体，对于谣言数据集的检测，可能会影响观感，因此注释了。
