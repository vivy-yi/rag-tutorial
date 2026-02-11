# 案例5：多模态产品问答

> **难度**: ⭐⭐⭐ 高级 | **技术栈**: CLIP, GPT-4V, 多模态Embedding, 图文检索

支持图片和文本混合检索的多模态RAG系统

---

## 🎯 案例特点

- ✅ **图文检索**: CLIP多模态嵌入
- ✅ **GPT-4V集成**: 视觉理解能力
- ✅ **混合查询**: 文本+图片联合检索
- ✅ **产品展示**: 电商场景优化

---

## 🚀 快速开始

```bash
cd projects/case5-multimodal
pip install -r requirements.txt
python main.py
```

---

## 🔑 核心技术

### CLIP多模态嵌入

```python
import clip
from PIL import Image

class MultiModalEmbedding:
    def __init__(self):
        # 加载CLIP模型
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def embed_text(self, text):
        """文本嵌入"""
        tokens = clip.tokenize([text])
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features

    def embed_image(self, image_path):
        """图像嵌入"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def similarity(self, text, image_path):
        """计算图文相似度"""
        text_feat = self.embed_text(text)
        image_feat = self.embed_image(image_path)
        return cosine_similarity(text_feat, image_feat)
```

### GPT-4V理解

```python
from openai import OpenAI

def understand_image(image_path, question):
    """使用GPT-4V理解图片"""
    client = OpenAI()

    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 调用GPT-4V
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"}
                }
            ]
        }]
    )

    return response.choices[0].message.content
```

---

## 📊 应用场景

### 场景1: 以图搜图
用户上传产品图 → 系统检索相似产品 → 返回推荐列表

### 场景2: 图文混合查询
"找一个红色连衣裙，款式和这张图片类似" → 结合视觉特征和文本描述

### 场景3: 视觉问答
上传产品照片 → "这个产品有什么特点？" → GPT-4V分析并回答

---

## 🎓 学习要点

1. **多模态嵌入**
   - CLIP模型原理
   - 图文对齐
   - 联合向量空间

2. **视觉理解**
   - GPT-4V应用
   - 图像描述生成
   - 视觉问答

3. **检索策略**
   - 多阶段检索
   - 特征融合
   - 相关性计算

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case5-multimodal)**

**[← 返回案例列表](index.md)**
