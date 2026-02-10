# 第16章：多模态RAG

> 突破文本限制：融合图像、视频、音频等多模态信息，构建真正的全感知RAG系统！

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解多模态嵌入模型
- [ ] 掌握CLIP等跨模态模型
- [ ] 实现图文检索RAG
- [ ] 构建多模态问答系统
- [ ] 了解多模态Agent实现

**预计学习时间**：6小时
**难度等级**：⭐⭐⭐⭐⭐

---

## 前置知识

- [ ] 完成模块1和模块2
- [ ] 了解多模态学习基础
- [ ] 熟悉OpenAI CLIP或类似模型
- [ ] 理解向量数据库

---

## 16.1 为什么需要多模态RAG？

### 16.1.1 单模态的局限

```
传统RAG（仅文本）：

用户查询："这张图片中的建筑是什么风格？"
问题：无法处理图片

系统：
  ✗ 无法理解图像内容
  ✗ 只能检索文本描述
  ✗ 答案质量受限

多模态RAG：

用户查询：[图片] + "这是什么风格？"

系统：
  ✓ 理解图像内容
  ✓ 检索相关图文信息
  ✓ 生成多模态答案
```

### 16.1.2 多模态应用场景

```
场景1：产品搜索
  输入：[产品图片]
  检索：相似产品（图像+文本）
  输出：产品推荐

场景2：医疗诊断
  输入：[X光片] + "有什么异常？"
  检索：相似病例 + 医学知识
  输出：诊断建议

场景3：教育问答
  输入：[题目截图] + "如何解答？"
  检索：类似题目 + 解题方法
  输出：详细解答

场景4：内容审核
  输入：[视频] + "是否违规？"
  检索：违规案例 + 审核规则
  输出：审核报告
```

---

## 16.2 多模态嵌入模型

### 16.2.1 CLIP模型

**CLIP (Contrastive Language-Image Pre-training)**

```
原理：
  图像编码器 + 文本编码器
  通过对比学习对齐

架构：
  Image Encoder (Vision Transformer)
    ↓
  Image Embedding (512维)

  Text Encoder (Transformer)
    ↓
  Text Embedding (512维)

  对比学习：对齐图像和文本嵌入
```

### 16.2.2 CLIP使用

```python
# 文件名：multimodal_embedding.py
"""
多模态嵌入：CLIP使用
"""

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util


# 加载CLIP模型
model = SentenceTransformer('clip-ViT-B-32')
print("CLIP模型已加载")

# 图像嵌入
def embed_image(image_path: str) -> torch.Tensor:
    """嵌入图像"""
    img = Image.open(image_path)
    img_emb = model.encode([img])
    return img_emb

# 文本嵌入
def embed_text(text: str) -> torch.Tensor:
    """嵌入文本"""
    text_emb = model.encode([text])
    return text_emb

# 计算相似度
def compute_similarity(image_emb: torch.Tensor,
                        text_emb: torch.Tensor) -> float:
    """计算图像-文本相似度"""
    # 使用余弦相似度
    similarity = util.cos_sim(image_emb, text_emb)[0][0]
    return float(similarity)


# 使用示例
if __name__ == "__main__":
    # 嵌入图像和文本
    image_emb = embed_image("./images/building.jpg")
    text_emb = embed_text("A modern skyscraper with glass facade")

    # 计算相似度
    similarity = compute_similarity(image_emb, text_emb)
    print(f"图像-文本相似度: {similarity:.4f}")

    # 批量文本检索
    texts = [
        "A historical building",
        "A modern office building",
        "A residential house"
    ]
    text_embeddings = model.encode(texts)

    similarities = util.cos_sim(image_emb, text_embeddings)[0]
    best_idx = similarities.argmax()

    print(f"\n最佳匹配: {texts[best_idx]} (相似度: {similarities[best_idx]:.4f})")
```

---

## 16.3 图文检索RAG

### 16.3.1 系统架构

```
图文检索RAG架构

┌─────────────────────────────────────────┐
│          多模态查询                      │
│  [图像] + [文本问题]                     │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │ 多模态嵌入  │
        └──────┬──────┘
               │
        ┌──────┴──────────┐
        │               │
    ┌───┴────┐      ┌───┴────────┐
    │图像检索│      │文本检索    │
    │(CLIP)  │      │(BM25/向量)│
    └───┬────┘      └───┬────────┘
        │               │
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │   结果融合    │
        └───────┬───────┘
                │
        ┌───────┴──────────┐
        │  LLM生成答案    │
        │  (图文结合)    │
        └─────────────────┘
```

### 16.3.2 完整实现

```python
# 文件名：multimodal_rag.py
"""
多模态RAG系统
"""

from typing import List, Dict, Union, Any
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


class MultiModalRAG:
    """
    多模态RAG系统

    支持图像+文本查询
    """

    def __init__(self,
                 image_db: List[str],
                 text_db: List[str],
                 llm_client):
        """
        初始化

        Args:
            image_db: 图像数据库路径列表
            text_db: 文本数据库
            llm_client: LLM客户端
        """
        # 加载CLIP模型
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

        # 预嵌入图像
        print("预嵌入图像...")
        self.image_paths = image_db
        self.image_embeddings = self.clip_model.encode(
            [Image.open(p) for p in image_db]
        )

        # 预嵌入文本
        print("预嵌入文本...")
        self.text_db = text_db
        self.text_embeddings = self.clip_model.encode(text_db)

        self.llm = llm_client

    def retrieve_by_image(self, image_path: str,
                          top_k: int = 5) -> List[Dict]:
        """
        图像检索：用图像查询

        Args:
            image_path: 查询图像路径
            top_k: 返回结果数
        """
        # 嵌入查询图像
        query_img = Image.open(image_path)
        query_emb = self.clip_model.encode([query_img])

        # 计算相似度
        similarities = util.cos_sim(query_emb, self.image_embeddings)[0]

        # Top-K
        top_k_indices = similarities.argsort()[::-1][:top_k]

        results = [
            {
                'image_path': self.image_paths[idx],
                'similarity': float(similarities[idx]),
                'type': 'image'
            }
            for idx in top_k_indices
        ]

        return results

    def retrieve_by_text(self, query: str,
                         top_k: int = 5) -> List[Dict]:
        """
        文本检索：用文本查询

        Args:
            query: 查询文本
            top_k: 返回结果数
        """
        # 嵌入查询文本
        query_emb = self.clip_model.encode([query])

        # 计算相似度
        similarities = util.cos_sim(query_emb, self.text_embeddings)[0]

        # Top-K
        top_k_indices = similarities.argsort()[::-1][:top_k]

        results = [
            {
                'text': self.text_db[idx],
                'similarity': float(similarities[idx]),
                'type': 'text'
            }
            for idx in top_k_indices
        ]

        return results

    def retrieve_multimodal(self,
                            query_text: str,
                            query_image: str = None,
                            top_k: int = 10) -> List[Dict]:
        """
        多模态检索：图像+文本查询

        Args:
            query_text: 查询文本
            query_image: 查询图像路径（可选）
            top_k: 返回结果数
        """
        results = []

        # 文本检索
        text_results = self.retrieve_by_text(query_text, top_k=top_k // 2)
        results.extend(text_results)

        # 图像检索（如果提供）
        if query_image:
            image_results = self.retrieve_by_image(query_image, top_k=top_k // 2)
            results.extend(image_results)

        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

    def generate_answer(self, query_text: str,
                       query_image: str = None,
                       context: List[Dict] = None) -> Dict:
        """
        生成多模态答案

        Args:
            query_text: 查询文本
            query_image: 查询图像（可选）
            context: 检索上下文
        """
        # 构建提示
        if query_image:
            # 包含图像的查询
            prompt = f"""基于以下图像和文本信息，回答问题。

问题：{query_text}

相关信息：
{self._format_context(context)}

请提供详细的答案。
"""
        else:
            # 纯文本查询
            prompt = f"""问题：{query_text}

相关信息：
{self._format_context(context)}

请提供详细的答案。
"""

        # 调用LLM生成答案
        answer = self.llm.generate(prompt)

        return {
            'answer': answer,
            'query_type': 'multimodal' if query_image else 'text',
            'context_used': context
        }

    def _format_context(self, context: List[Dict]) -> str:
        """格式化上下文"""
        if not context:
            return "无相关信息"

        formatted = []
        for i, item in enumerate(context[:5], 1):
            if item['type'] == 'image':
                formatted.append(f"{i}. [图像] {item['image_path']} (相似度: {item['similarity']:.2f})")
            else:
                formatted.append(f"{i}. [文本] {item['text'][:100]}... (相似度: {item['similarity']:.2f})")

        return "\n".join(formatted)


# 使用示例
if __name__ == "__main__":
    # 准备数据
    image_db = [
        "./data/images/building1.jpg",
        "./data/images/building2.jpg",
        "./data/images/building3.jpg"
    ]

    text_db = [
        "现代建筑风格包括玻璃幕墙、流线型设计等",
        "哥特式建筑特点是尖顶、飞扶壁、彩色玻璃",
        "中国古代建筑以木结构为主，如斗拱、榫卯"
    ]

    # 创建多模态RAG系统
    system = MultiModalRAG(image_db, text_db, llm_client=None)

    # 图文查询
    results = system.retrieve_multimodal(
        query_text="现代建筑风格特点",
        query_image="./data/images/query.jpg",
        top_k=5
    )

    print("多模态检索结果:")
    for r in results:
        if r['type'] == 'image':
            print(f"图像: {r['image_path']} (相似度: {r['similarity']:.2f})")
        else:
            print(f"文本: {r['text'][:50]}... (相似度: {r['similarity']:.2f})")
```

---

## 16.4 多模态Agent

### 16.4.1 多模态工具定义

```python
# 文件名：multimodal_agent.py
"""
多模态Agent
"""

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.tools import StructuredTool


class MultiModalAgent:
    """
    多模态Agent

    支持处理图像、文本、音频等多种模态
    """

    def __init__(self, openai_api_key: str):
        from langchain_openai import ChatOpenAI
        from sentence_transformers import SentenceTransformer

        # LLM
        self.llm = ChatOpenAI(model="gpt-4-vision-preview", api_key=openai_api_key)

        # CLIP模型
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

        # 定义工具
        self.tools = self._create_tools()

        # 创建Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )

    def _create_tools(self) -> List[Tool]:
        """创建多模态工具"""
        tools = [
            Tool(
                name="ImageSearch",
                func=self._image_search,
                description="搜索相似的图像。输入：图像路径或描述。"
            ),
            Tool(
                name="TextSearch",
                func=self._text_search,
                description="搜索相关文档。输入：搜索关键词。"
            ),
            Tool(
                name="ImageAnalysis",
                func=self._image_analysis,
                description="分析图像内容。输入：图像路径。"
            ),
            Tool(
                name="MultimodalQA",
                func=self._multimodal_qa,
                description="基于图像和文本回答问题。输入：图像路径和问题。"
            )
        ]
        return tools

    def _image_search(self, query: str) -> str:
        """图像搜索"""
        # 实际实现搜索图像数据库
        return f"找到与'{query}'相似的5张图像..."

    def _text_search(self, query: str) -> str:
        """文本搜索"""
        # 实际实现搜索文档数据库
        return f"找到与'{query}'相关的10个文档..."

    def _image_analysis(self, image_path: str) -> str:
        """图像分析"""
        # 使用GPT-4V分析图像
        from PIL import Image

        image = Image.open(image_path)

        # 调用GPT-4V
        response = self.llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片的内容。"},
                    {"type": "image_url", "image_url": image_path}
                ]
            }
        ])

        return response.content[0].text

    def _multimodal_qa(self, inputs: str) -> str:
        """多模态问答"""
        # 解析输入（格式："image_path: 问题"）
        parts = inputs.split(":")
        image_path = parts[0].strip()
        question = parts[1].strip() if len(parts) > 1 else "这张图片是什么？"

        # 图像+文本多模态理解
        from PIL import Image

        image = Image.open(image_path)

        response = self.llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"回答这个问题：{question}"},
                    {"type": "image_url", "image_url": image_path}
                ]
            }
        ])

        return response.content[0].text

    def query(self, user_input: Union[str, tuple]) -> Dict:
        """
        多模态查询

        Args:
            user_input: 文本查询 或 (image_path, query)

        Returns:
            查询结果
        """
        # 转换输入格式
        if isinstance(user_input, tuple):
            # (图像路径, 问题)
            query_str = f"{user_input[0]}: {user_input[1]}"
        else:
            query_str = user_input

        # 执行Agent
        result = self.executor.invoke({"input": query_str})

        return {
            'answer': result['output'],
            'intermediate_steps': result.get('intermediate_steps', [])
        }


# 使用示例
if __name__ == "__main__":
    import os

    agent = MultiModalAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # 图像查询
    result1 = agent.query(("./images/photo.jpg", "这张照片是什么风格？"))
    print(f"答案: {result1['answer']}")

    # 纯文本查询
    result2 = agent.query("比较Python和JavaScript的性能")
    print(f"答案: {result2['answer']}")
```

---

## 16.5 完整项目：图文问答系统

### 16.5.1 项目设计

**项目名称**：VisualQA System

**功能需求**：
1. 图像上传和识别
2. 图像检索
3. 图文结合的问答
4. 多模态答案生成
5. Web界面（Streamlit）

### 16.5.2 完整实现

```python
# 文件名：visual_qa_app.py
"""
图文问答系统Web应用
"""

import streamlit as st
from PIL import Image
from multimodal_rag import MultiModalRAG
import os


def main():
    st.set_page_config(page_title="多模态RAG问答系统", layout="wide")

    st.title("🖼️📝 多模态RAG问答系统")
    st.markdown("支持图像+文本查询，提供智能问答服务")

    # 侧边栏
    with st.sidebar:
        st.header("设置")
        api_key = st.text_input("OpenAI API Key", type="password")

        if api_key:
            st.session_state.api_key = api_key

    # 主界面
    tab1, tab2, tab3 = st.tabs(["图文问答", "图像检索", "系统说明"])

    with tab1:
        st.header("图文问答")

        # 上传图像
        uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_column_width=True)

            # 输入问题
            question = st.text_input("请输入您的问题...")

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("🚀 提交", use_container_width=True):
                    if question and api_key:
                        with st.spinner("正在分析..."):
                            # 调用多模态RAG
                            system = MultiModalRag(api_key)
                            result = system.query((uploaded_file, question))

                            st.subheader("答案")
                            st.write(result['answer'])
                    else:
                        st.warning("请输入问题")
```

---

## 练习题

### 练习16.1：CLIP图像检索（基础）

**题目**：使用CLIP实现图像检索系统

**要求**：
1. 准备100张图像数据集
2. 使用CLIP嵌入图像
3. 实现相似图像检索
4. 提供可视化结果

---

### 练习16.2：图文RAG实现（进阶）

**题目**：构建图文结合的RAG系统

**要求**：
1. 图像检索模块
2. 文本检索模块
3. 结果融合算法
4. 多模态答案生成
5. 评估系统效果

---

### 练习16.3：多模态Agent（挑战）

**题目**：实现能够理解图像和文本的Agent

**功能需求**：
1. 图像理解和分析
2. 图文结合的推理
3. 多工具调用
4. 复杂任务处理
5. Web界面

---

## 总结

### 本章要点

1. **多模态价值**
   - 突破单模态限制
   - 丰富的信息来源
   - 更好的用户体验

2. **CLIP模型**
   - 图像-文本对齐
   - 跨模态检索
   - 零样本能力

3. **多模态RAG**
   - 图像+文本嵌入
   - 跨模态检索
   - 多模态答案生成

4. **多模态Agent**
   - 多模态工具
   - GPT-4V集成
   - 复杂任务处理

### 学习检查清单

- [ ] 理解多模态嵌入
- [ ] 掌握CLIP使用
- [ ] 实现图文RAG
- [ ] 构建多模态系统

### 模块3完成！

**恭喜完成模块3的学习！** 🎉

你已经掌握了：
- ✅ Agentic RAG（第13章）
- ✅ 高级Agent模式（第14章）
- ✅ 知识图谱RAG（第15章）
- ✅ 多模态RAG（第16章）

**能力提升**：
- 复杂问题解决能力 +60%
- 系统自主性 +80%
- 知识整合能力 +70%
- 架构设计能力 +50%

### 下一步

**选项1**：开始模块4 - 生产部署实战
**选项2**：创建更多配套资源
**选项3**：开始综合项目

---

**模块3完成！** 🎊🎊🎊
