# 案例5：多模态产品问答系统

> 构建能够理解和生成图像+文本的电商产品问答系统

---

## 📋 案例概述

### 业务场景

电商平台产品咨询挑战：
- ✗ 用户只能用文字描述产品
- ✗ 难以找到相似产品图片
- ✗ 视觉特性无法文字表达
- ✗ 产品对比耗时

### 多模态RAG解决方案

构建图文结合的产品问答系统：
- ✅ 图像上传识别
- ✅ 图文混合检索
- ✅ 视觉问答
- ✅ 智能推荐

---

## 🎯 功能需求

### 核心功能

1. **图像理解**
   - 上传产品图片
   - 自动识别产品类型
   - 提取视觉特征
   - 生成描述

2. **图文检索**
   - 以图搜图
   - 以文搜图
   - 图文混合搜索
   - 相似产品推荐

3. **视觉问答**
   - 回答产品相关问题
   - 解释产品特性
   - 对比多个产品
   - 提供购买建议

4. **智能推荐**
   - 基于视觉相似推荐
   - 基于文本描述推荐
   - 个性化推荐
   - 跨类别推荐

---

## 🏗️ 系统架构

### 架构图

```
┌──────────────────────────────────────────┐
│         Streamlit Web界面                │
│  - 图像上传                              │
│  - 问答交互                               │
│  - 结果展示                               │
└──────────────┬───────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────┐
│          FastAPI后端                      │
│  - 多模态API                              │
│  - 结果融合                               │
│  - 业务逻辑                               │
└───────┬────────────────┬─────────────────┘
        │                │
    ┌───┴────┐      ┌───┴────────┐
    │CLIP模型│      │  GPT-4V    │
    │(嵌入)  │      │  (理解)    │
    └────────┘      └────────────┘
        │                │
        └────────┬───────┘
                 ↓
        ┌──────────────┐
        │ 向量数据库    │
        │ (ChromaDB)   │
        └──────────────┘
```

### 技术栈

**多模态模型**：
- CLIP (图像-文本嵌入)
- GPT-4V (视觉理解)
- BLIP (图像描述生成)

**向量数据库**：
- ChromaDB
- Pinecone (可选)

**后端**：
- FastAPI
- OpenAI API

**前端**：
- Streamlit
- Pillow (图像处理)

---

## 💻 核心实现

### 1. 多模态嵌入

```python
# multimodal_embedding.py
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
from typing import List, Union, Tuple

class MultiModalEmbedding:
    """
    多模态嵌入服务

    使用CLIP实现图像-文本联合嵌入
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        初始化CLIP模型

        Args:
            model_name: 模型名称
        """
        print(f"加载CLIP模型: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> torch.Tensor:
        """
        嵌入图像

        Args:
            image_path: 图像路径

        Returns:
            图像嵌入向量
        """
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def embed_text(self, text: str) -> torch.Tensor:
        """
        嵌入文本

        Args:
            text: 文本内容

        Returns:
            文本嵌入向量
        """
        embedding = self.model.encode([text])
        return embedding[0]

    def embed_batch_images(self,
                          image_paths: List[str]) -> torch.Tensor:
        """
        批量嵌入图像

        Args:
            image_paths: 图像路径列表

        Returns:
            图像嵌入矩阵
        """
        images = [Image.open(path) for path in image_paths]
        embeddings = self.model.encode(images)
        return embeddings

    def compute_similarity(self,
                         image_emb: torch.Tensor,
                         text_emb: torch.Tensor) -> float:
        """
        计算图像-文本相似度

        Args:
            image_emb: 图像嵌入
            text_emb: 文本嵌入

        Returns:
            相似度分数 (0-1)
        """
        similarity = util.cos_sim(
            image_emb.reshape(1, -1),
            text_emb.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def find_similar_images(self,
                           query_image: str,
                           image_paths: List[str],
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找相似图像

        Args:
            query_image: 查询图像
            image_paths: 候选图像列表
            top_k: 返回前K个结果

        Returns:
            [(图像路径, 相似度)]
        """
        # 嵌入查询图像
        query_emb = self.embed_image(query_image)

        # 批量嵌入候选图像
        candidate_embs = self.embed_batch_images(image_paths)

        # 计算相似度
        similarities = util.cos_sim(
            query_emb.reshape(1, -1),
            candidate_embs
        )[0]

        # Top-K
        top_k_indices = similarities.argsort(descending=True)[:top_k]

        results = [
            (image_paths[i], float(similarities[i]))
            for i in top_k_indices
        ]

        return results
```

### 2. 产品索引

```python
# product_indexer.py
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from multimodal_embedding import MultiModalEmbedding

class ProductIndexer:
    """
    产品索引器

    索引产品的图像和文本信息
    """

    def __init__(self,
                 persist_directory: str = "./data/chroma"):
        """
        初始化

        Args:
            persist_directory: ChromaDB持久化目录
        """
        # 初始化向量数据库
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 创建或获取collection
        self.collection = self.client.get_or_create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"}
        )

        # 初始化嵌入模型
        self.embedder = MultiModalEmbedding()

    def index_products(self, products: List[Dict]):
        """
        索引产品

        Args:
            products: 产品列表
            [{
                "id": "P001",
                "name": "无线耳机",
                "description": "蓝牙5.0，降噪功能",
                "image_path": "products/P001.jpg",
                "category": "电子产品",
                "price": 299
            }]
        """
        for product in products:
            try:
                # 嵌入图像
                image_emb = self.embedder.embed_image(product["image_path"])

                # 嵌入文本（名称+描述）
                text = f"{product['name']} {product['description']}"
                text_emb = self.embedder.embed_text(text)

                # 合并嵌入（简单平均）
                combined_emb = (image_emb + text_emb) / 2

                # 添加到向量库
                self.collection.add(
                    embeddings=[combined_emb.tolist()],
                    documents=[text],
                    metadatas=[{
                        "id": product["id"],
                        "name": product["name"],
                        "category": product["category"],
                        "price": product["price"],
                        "image_path": product["image_path"]
                    }],
                    ids=[product["id"]]
                )

                print(f"✓ 索引产品: {product['name']}")

            except Exception as e:
                print(f"✗ 索引失败 {product['id']}: {e}")

    def search_by_image(self,
                       query_image: str,
                       top_k: int = 5) -> List[Dict]:
        """
        以图搜图

        Args:
            query_image: 查询图像
            top_k: 返回结果数

        Returns:
            相似产品列表
        """
        # 嵌入查询图像
        query_emb = self.embedder.embed_image(query_image)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        # 格式化结果
        products = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0],
            results["metadatas"][0])
        ):
            products.append({
                "rank": i + 1,
                "id": metadata["id"],
                "name": metadata["name"],
                "category": metadata["category"],
                "price": metadata["price"],
                "image_path": metadata["image_path"],
                "description": doc
            })

        return products

    def search_by_text(self,
                       query_text: str,
                       top_k: int = 5) -> List[Dict]:
        """
        文本搜索产品

        Args:
            query_text: 查询文本
            top_k: 返回结果数

        Returns:
            相关产品列表
        """
        # 嵌入查询文本
        query_emb = self.embedder.embed_text(query_text)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        # 格式化结果
        products = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0],
            results["metadatas"][0])
        ):
            products.append({
                "rank": i + 1,
                "id": metadata["id"],
                "name": metadata["name"],
                "category": metadata["category"],
                "price": metadata["price"],
                "image_path": metadata["image_path"],
                "description": doc
            })

        return products

    def multimodal_search(self,
                         query_text: str = None,
                         query_image: str = None,
                         top_k: int = 5) -> List[Dict]:
        """
        多模态搜索

        Args:
            query_text: 查询文本（可选）
            query_image: 查询图像（可选）
            top_k: 返回结果数

        Returns:
            融合结果
        """
        if query_image and query_text:
            # 图文融合
            image_emb = self.embedder.embed_image(query_image)
            text_emb = self.embedder.embed_text(query_text)

            # 加权融合
            combined_emb = (image_emb + text_emb) / 2

            results = self.collection.query(
                query_embeddings=[combined_emb.tolist()],
                n_results=top_k
            )

        elif query_image:
            # 纯图像
            return self.search_by_image(query_image, top_k)

        elif query_text:
            # 纯文本
            return self.search_by_text(query_text, top_k)

        else:
            return []

        # 格式化结果
        products = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0],
            results["metadatas"][0])
        ):
            products.append({
                "rank": i + 1,
                "id": metadata["id"],
                "name": metadata["name"],
                "category": metadata["category"],
                "price": metadata["price"],
                "image_path": metadata["image_path"],
                "description": doc
            })

        return products
```

### 3. 视觉问答

```python
# visual_qa.py
from openai import OpenAI
from PIL import Image
import base64
from typing import Dict

class VisualQuestionAnswering:
    """
    视觉问答服务

    使用GPT-4V理解图像并回答问题
    """

    def __init__(self, api_key: str):
        """
        初始化

        Args:
            api_key: OpenAI API密钥
        """
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image_path: str) -> str:
        """
        编码图像为base64

        Args:
            image_path: 图像路径

        Returns:
            base64编码的图像
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(
                image_file.read()
            ).decode('utf-8')

    def answer_question(self,
                       image_path: str,
                       question: str) -> Dict:
        """
        回答关于图像的问题

        Args:
            image_path: 图像路径
            question: 问题文本

        Returns:
            答案
        """
        # 编码图像
        base64_image = self.encode_image(image_path)

        # 调用GPT-4V
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "image_path": image_path
        }

    def compare_products(self,
                        image1: str,
                        image2: str) -> Dict:
        """
        对比两个产品

        Args:
            image1: 产品1图像
            image2: 产品2图像

        Returns:
            对比结果
        """
        base64_1 = self.encode_image(image1)
        base64_2 = self.encode_image(image2)

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "对比这两个产品，分析它们的异同点、优缺点。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_1}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_2}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        comparison = response.choices[0].message.content

        return {
            "image1": image1,
            "image2": image2,
            "comparison": comparison
        }

    def generate_description(self, image_path: str) -> str:
        """
        生成产品描述

        Args:
            image_path: 产品图像

        Returns:
            产品描述
        """
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """请详细描述这个产品，包括：
1. 产品类型
2. 主要特征
3. 设计风格
4. 目标用户
5. 卖点"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content
```

### 4. Streamlit界面

```python
# app.py
import streamlit as st
from PIL import Image
import os

# 页面配置
st.set_page_config(
    page_title="多模态产品问答",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️📝 多模态产品问答系统")
st.markdown("上传产品图片，智能问答和推荐")

# 侧边栏
with st.sidebar:
    st.header("功能选择")

    mode = st.radio(
        "选择模式",
        ["📸 图像检索", "💬 视觉问答", "🔄 产品对比", "✨ 智能推荐"]
    )

    st.divider()

    st.subheader("设置")
    top_k = st.slider("推荐数量", 1, 10, 5)
    api_key = st.text_input("OpenAI API Key", type="password")

# 主界面
if mode == "📸 图像检索":
    st.header("图像检索")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上传查询图像")
        uploaded_file = st.file_uploader(
            "选择产品图片",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file:
            # 保存临时文件
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open(temp_path)
            st.image(image, caption="上传的图片", use_column_width=True)

            query_text = st.text_input(
                "可选：添加文字描述"
            )

            if st.button("🔍 搜索相似产品", type="primary"):
                # 执行检索（需要后端）
                with st.spinner("正在搜索..."):
                    # 这里调用后端API
                    # results = indexer.multimodal_search(
                    #     query_image=temp_path,
                    #     query_text=query_text or None,
                    #     top_k=top_k
                    # )

                    # 模拟结果
                    st.session_state.results = [
                        {
                            "name": f"产品{i}",
                            "price": f"{100 + i*50}元",
                            "similarity": 0.95 - i*0.05
                        }
                        for i in range(top_k)
                    ]

    with col2:
        if 'results' in st.session_state:
            st.subheader(f"找到 {len(st.session_state.results)} 个相似产品")

            for i, product in enumerate(st.session_state.results, 1):
                with st.expander(
                    f"{i}. {product['name']} - {product['price']}",
                    expanded=(i == 1)
                ):
                    st.write(f"**相似度**: {product['similarity']:.2%}")
                    # 显示产品图片
                    # st.image(product['image_path'])

elif mode == "💬 视觉问答":
    st.header("视觉问答")

    uploaded_file = st.file_uploader(
        "上传产品图片",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="产品图片", use_column_width=True)

        with col2:
            question = st.text_area(
                "输入问题",
                placeholder="例如：这个产品有什么特点？",
                height=100
            )

            if st.button("❓ 提问", type="primary"):
                if not api_key:
                    st.error("请输入OpenAI API Key")
                elif not question:
                    st.warning("请输入问题")
                else:
                    with st.spinner("正在分析..."):
                        # 保存临时文件
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # 调用视觉问答
                        # vqa = VisualQuestionAnswering(api_key)
                        # result = vqa.answer_question(temp_path, question)
                        # st.write(result['answer'])

                        # 模拟
                        st.write("""
                        这个产品是一款**无线耳机**，具有以下特点：

                        **主要特征**：
                        - 蓝牙5.0连接，低延迟
                        - 主动降噪功能
                        - 30小时续航
                        - IPX4防水

                        **设计风格**：简约现代，人体工学

                        **目标用户**：音乐爱好者、商务人士

                        **卖点**：高性价比、长续航、舒适佩戴
                        """)

elif mode == "🔄 产品对比":
    st.header("产品对比")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("产品1")
        file1 = st.file_uploader("上传产品1", type=['jpg', 'png'])

    with col2:
        st.subheader("产品2")
        file2 = st.file_uploader("上传产品2", type=['jpg', 'png'])

    if file1 and file2:
        if st.button("🔄 对比产品", type="primary"):
            with st.spinner("正在分析..."):
                # 调用对比API
                st.session_state.comparison = "对比结果..."

    if 'comparison' in st.session_state:
        st.subheader("对比结果")
        st.write(st.session_state.comparison)

elif mode == "✨ 智能推荐":
    st.header("智能推荐")

    st.markdown("""
    基于您的上传图片，推荐以下产品：

    1. **相似产品推荐**
       - 基于视觉特征相似度
       - 相同类别产品
       - 同品牌产品

    2. **互补产品推荐**
       - 配套产品
       - 相关配件
       - 使用场景建议

    3. **升级产品推荐**
       - 更高端型号
       - 新版本
       - 替代产品
    """)
```

---

## 📊 数据准备

### 产品数据格式

```json
[
  {
    "id": "P001",
    "name": "Sony WH-1000XM4 无线耳机",
    "category": "电子产品",
    "price": 2299,
    "description": "行业领先的降噪耳机，30小时续航",
    "image_path": "products/headphones_sony.jpg",
    "brand": "Sony",
    "features": ["降噪", "无线", "长续航"],
    "tags": ["耳机", "音频", "蓝牙"]
  },
  {
    "id": "P002",
    "name": "AirPods Pro 2",
    "category": "电子产品",
    "price": 1899,
    "description": "Apple主动降噪耳机，空间音频",
    "image_path": "products/headphones_apple.jpg",
    "brand": "Apple",
    "features": ["降噪", "空间音频", "通透模式"],
    "tags": ["耳机", "音频", "iOS"]
  }
]
```

---

## 🧪 测试场景

### 场景1：以图搜图

**输入**：上传一张耳机图片

**输出**：
1. Sony WH-1000XM4 (相似度: 95%)
2. Bose QuietComfort 45 (相似度: 89%)
3. AirPods Pro 2 (相似度: 85%)

### 场景2：视觉问答

**输入**：
- 图片：产品图
- 问题："这个产品的续航时间是多少？"

**输出**：
"根据产品信息，这款耳机提供**30小时**的总续航时间（开启降噪模式下为20小时），支持快充，充电10分钟可使用5小时。"

### 场景3：图文混合检索

**输入**：
- 图片：产品图
- 文本："价格在2000元以下"

**输出**：符合视觉特征和价格限制的产品

---

## 📈 性能优化

### 1. 批量嵌入

```python
def batch_index_products(products: List[Dict],
                         batch_size: int = 32):
    """批量索引产品"""
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        # 批量嵌入
        embeddings = embedder.encode(
            [p['image_path'] for p in batch],
            batch_size=batch_size,
            show_progress_bar=False
        )
        # 批量添加
        collection.add(embeddings=embeddings, ...)
```

### 2. 图像预处理

```python
def preprocess_image(image_path: str,
                     size: int = 224) -> Image.Image:
    """预处理图像"""
    img = Image.open(image_path)

    # 调整大小
    img = img.resize((size, size))

    # 归一化
    # img = normalize(img)

    return img
```

### 3. 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_image_embedding(image_path: str):
    """缓存图像嵌入"""
    return embedder.embed_image(image_path)
```

---

## 🎨 UI优化

### 1. 图像预览

```python
def display_product_grid(products: List[Dict],
                         columns: int = 4):
    """展示产品网格"""
    for i in range(0, len(products), columns):
        cols = st.columns(columns)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(products):
                with col:
                    st.image(products[idx]['image_path'])
                    st.write(products[idx]['name'])
                    st.caption(f"¥{products[idx]['price']}")
```

### 2. 相似度可视化

```python
import plotly.graph_objects as go

def plot_similarity_bar(products: List[Dict]):
    """绘制相似度条形图"""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[p['similarity'] for p in products],
                y=[p['name'] for p in products],
                orientation='h',
                marker_color='skyblue'
            )
        ]
    )

    fig.update_layout(
        title="产品相似度",
        xaxis_title="相似度",
        yaxis_title="产品",
        height=400
    )

    st.plotly_chart(fig)
```

---

## 🎓 学习要点

完成本案例后，你将掌握：

### ✅ 多模态技术
- CLIP模型使用
- 图像-文本嵌入
- 跨模态检索
- 融合策略

### ✅ 视觉问答
- GPT-4V集成
- 图像理解
- 视觉推理
- 答案生成

### ✅ 电商应用
- 产品索引
- 相似推荐
- 视觉搜索
- 智能对比

---

## 🚀 进阶方向

1. **高级功能**
   - 视频理解
   - 3D产品展示
   - AR试用
   - 个性化推荐

2. **性能提升**
   - 分布式检索
   - 图像压缩
   - 增量索引
   - 实时更新

3. **商业价值**
   - 用户行为分析
   - A/B测试
   - 转化率优化
   - 精准营销

---

## 📚 参考资源

- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [CLIP论文](https://openai.com/research/clip/)
- [Sentence-Transformers](https://www.sbert.net/examples/applications/multilingual-semantic.html)

---

**开始构建你的多模态产品问答系统吧！** 🚀
