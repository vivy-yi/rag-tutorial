# 模块1练习题与答案

## 第1章：RAG技术概述

### 练习1.1：概念理解

**题目**：判断以下说法是否正确，并说明理由

1. RAG可以完全解决LLM的幻觉问题
2. RAG系统必须使用向量数据库
3. Fine-tuning可以替代RAG
4. RAG更适合知识频繁更新的场景

**答案**：

1. ❌ 错误。RAG可以显著减少但不能完全解决幻觉，还需要依赖检索质量和LLM生成能力。
2. ❌ 错误。虽然向量数据库是常见选择，但也可以使用其他检索方法（如BM25）。
3. ❌ 错误。两者各有优势，RAG处理知识更新，Fine-tuning学习特定风格和格式，可以结合使用。
4. ✅ 正确。RAG无需重新训练即可更新知识，非常适合频繁变化的场景。

---

### 练习1.2：组件匹配

**题目**：将以下功能与对应的组件匹配

功能 → 组件
1. 从PDF文件中提取文本 → ?
2. 将"人工智能很强大"转换为向量 → ?
3. 在100万个向量中快速搜索 → ?
4. 将长文档切成500字的块 → ?
5. 基于检索文档生成流畅答案 → ?

**答案**：
1. → 文档加载器
2. → 嵌入模型
3. → 向量数据库
4. → 文本分块器
5. → 大语言模型（LLM）

---

### 练习1.3：场景分析

**题目**：以下场景应该使用RAG、Fine-tuning还是Prompt Engineering？

1. 让AI严格按照JSON格式输出
2. 回答公司2024年的销售数据
3. 提升AI的代码生成能力
4. 回答用户关于产品手册的问题
5. 让AI用专业的法律文书风格写作

**答案**：

1. **Prompt Engineering** - 格式要求，不需要外部知识
2. **RAG** - 需要准确的公司内部数据
3. **Fine-tuning** - 提升通用能力
4. **RAG** - 基于具体产品文档
5. **Fine-tuning** - 学习特定风格

---

## 第2章：环境搭建与工具准备

### 练习2.1：环境配置

**题目**：完成以下环境配置任务

```python
# 任务1：检查Python版本
import sys
# TODO: 打印Python版本，检查是否>=3.9

# 任务2：创建虚拟环境
# TODO: 写出创建虚拟环境的命令

# 任务3：安装LlamaIndex
# TODO: 写出安装命令
```

**答案**：

```python
# 任务1
import sys
print(f"Python版本: {sys.version}")
# 检查
if sys.version_info >= (3, 9):
    print("✅ 版本符合要求")
else:
    print("❌ 需要升级到Python 3.9+")

# 任务2
# 命令行：
# python -m venv rag_env
# source rag_env/bin/activate  # Linux/Mac
# rag_env\Scripts\activate     # Windows

# 任务3
# pip install llama-index-core
# pip install llama-index-llms-openai
# pip install llama-index-embeddings-openai
```

---

### 练习2.2：依赖安装

**题目**：创建一个requirements.txt文件，包含RAG系统的所有必要依赖

**答案**：

```txt
# RAG系统依赖

# 核心框架
llama-index-core>=0.10.0
llama-index-llms-openai>=0.1.0
llama-index-embeddings-openai>=0.1.0
llama-index-vector-stores-chroma>=0.1.0

# 向量数据库
chromadb>=0.4.0

# 文档处理
pypdf>=3.0.0
docx2txt>=0.8
python-dotenv>=1.0.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 工具
tqdm>=4.65.0
rich>=13.0.0
```

---

## 第3章：基础RAG实现

### 练习3.1：文档加载

**题目**：编写代码加载指定目录下的所有TXT文件

```python
from llama_index.core import SimpleDirectoryReader

def load_txt_files(directory_path):
    """
    加载目录中的所有TXT文件

    Args:
        directory_path: 目录路径

    Returns:
        文档列表
    """
    # TODO: 实现加载逻辑
    pass
```

**答案**：

```python
from llama_index.core import SimpleDirectoryReader

def load_txt_files(directory_path):
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".txt"],  # 只加载TXT文件
        recursive=True  # 递归加载子目录
    )
    documents = reader.load_data()
    return documents

# 使用示例
docs = load_txt_files("data/processed")
print(f"加载了 {len(docs)} 个文档")
```

---

### 练习3.2：文本分块

**题目**：实现一个分块函数，按段落切分文档

```python
def split_by_paragraph(text, max_chunk_size=1000):
    """
    按段落分块

    Args:
        text: 输入文本
        max_chunk_size: 最大块大小（字符数）

    Returns:
        分块列表
    """
    # TODO: 实现分块逻辑
    pass
```

**答案**：

```python
def split_by_paragraph(text, max_chunk_size=1000):
    # 按段落分割
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果当前块加上新段落超过限制
        if len(current_chunk) + len(para) > max_chunk_size:
            if current_chunk:  # 保存当前块
                chunks.append(current_chunk.strip())
                current_chunk = para  # 开始新块
            else:
                # 单个段落太长，强制切分
                chunks.append(para[:max_chunk_size])
                current_chunk = para[max_chunk_size:]
        else:
            # 累积到当前块
            current_chunk += "\n\n" + para if current_chunk else para

    # 保存最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 测试
sample_text = """
第一段内容。

第二段内容，这里有一些文字。

第三段内容，这是最后一段。
"""

chunks = split_by_paragraph(sample_text)
for i, chunk in enumerate(chunks, 1):
    print(f"块{i} ({len(chunk)}字符): {chunk[:50]}...")
```

---

### 练习3.3：向量检索

**题目**：实现简单的余弦相似度计算

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    Args:
        vec1, vec2: 向量

    Returns:
        相似度分数（0-1）
    """
    # TODO: 实现计算逻辑
    pass
```

**答案**：

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    余弦相似度 = (A·B) / (||A|| * ||B||)
    """
    # 点积
    dot_product = np.dot(vec1, vec2)

    # 向量长度（L2范数）
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 余弦相似度
    similarity = dot_product / (norm1 * norm2)

    return similarity

# 测试
vec1 = np.array([1, 2, 3])
vec2 = np.array([1, 2, 3])
vec3 = np.array([3, 2, 1])

print(f"相同向量相似度: {cosine_similarity(vec1, vec2):.3f}")  # 1.0
print(f"不同向量相似度: {cosine_similarity(vec1, vec3):.3f}")  # < 1.0
```

---

### 练习3.4：完整RAG流程

**题目**：实现一个简单的RAG查询函数

```python
def simple_rag_query(question, documents, embed_model, vector_store, llm):
    """
    简单的RAG查询

    Args:
        question: 用户问题
        documents: 文档列表
        embed_model: 嵌入模型
        vector_store: 向量数据库
        llm: 语言模型

    Returns:
        答案
    """
    # TODO: 实现完整流程
    # 1. 对问题进行嵌入
    # 2. 在向量库中检索
    # 3. 构建提示词
    # 4. 生成答案

    pass
```

**答案**：

```python
def simple_rag_query(question, documents, embed_model, vector_store, llm):
    # 步骤1：对问题进行嵌入
    query_embedding = embed_model.get_embedding(question)

    # 步骤2：检索相关文档
    retrieved_docs = vector_store.query(query_embedding, top_k=3)

    if not retrieved_docs:
        return "抱歉，知识库中没有找到相关信息。"

    # 步骤3：构建提示词
    context = "\n\n".join([f"文档{i+1}: {doc}"
                           for i, doc in enumerate(retrieved_docs)])

    prompt = f"""
基于以下文档回答用户问题。如果文档中没有相关信息，请明确说明。

参考文档：
{context}

用户问题：{question}

请提供准确的答案：
"""

    # 步骤4：生成答案
    answer = llm.generate(prompt)

    return answer
```

---

## 第4章：RAG评估基础

### 练习4.1：Hit Rate计算

**题目**：计算以下检索结果的Hit Rate

```python
queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]
retrieved_docs = [
    [1, 5, 8],      # Q1检索结果
    [2, 6, 9],      # Q2检索结果
    [3, 7, 10],     # Q3检索结果
    [4, 8, 12],     # Q4检索结果
    [5, 9, 13]      # Q5检索结果
]
relevant_docs = [
    {1, 8},         # Q1相关文档
    {6},            # Q2相关文档
    {11, 12},       # Q3相关文档
    {4},            # Q4相关文档
    {9}             # Q5相关文档
]

# TODO: 计算Hit Rate
```

**答案**：

```python
def calculate_hit_rate(retrieved_docs, relevant_docs):
    hits = 0
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        # 检查是否有至少一个相关文档
        if any(doc in relevant for doc in retrieved):
            hits += 1

    return hits / len(retrieved_docs)

hit_rate = calculate_hit_rate(retrieved_docs, relevant_docs)
print(f"Hit Rate: {hit_rate:.2%}")

# 结果分析：
# Q1: 检索到[1,5,8], 相关{1,8} → ✅ 命中
# Q2: 检索到[2,6,9], 相关{6} → ✅ 命中
# Q3: 检索到[3,7,10], 相关{11,12} → ❌ 未命中
# Q4: 检索到[4,8,12], 相关{4} → ✅ 命中
# Q5: 检索到[5,9,13], 相关{9} → ✅ 命中

# Hit Rate = 4/5 = 0.8 = 80%
```

---

### 练习4.2：MRR计算

**题目**：计算上述数据的MRR

**答案**：

```python
def calculate_mrr(retrieved_docs, relevant_docs):
    reciprocal_ranks = []

    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        # 找第一个相关文档的位置
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

mrr = calculate_mrr(retrieved_docs, relevant_docs)
print(f"MRR: {mrr:.3f}")

# 结果分析：
# Q1: 第一个相关文档是1，排第1 → 1/1 = 1.0
# Q2: 第一个相关文档是6，排第2 → 1/2 = 0.5
# Q3: 没有检索到相关文档 → 0.0
# Q4: 第一个相关文档是4，排第1 → 1/1 = 1.0
# Q5: 第一个相关文档是9，排第2 → 1/2 = 0.5

# MRR = (1.0 + 0.5 + 0.0 + 1.0 + 0.5) / 5 = 0.6
```

---

### 练习4.3：评估框架

**题目**：实现一个简单的评估器类

```python
class SimpleEvaluator:
    """简单的RAG评估器"""

    def __init__(self):
        self.metrics = {}

    def evaluate(self, questions, retrieved_docs, relevant_docs):
        """
        评估检索质量

        Args:
            questions: 问题列表
            retrieved_docs: 检索结果
            relevant_docs: 真实相关文档

        Returns:
            评估指标字典
        """
        # TODO: 实现评估逻辑
        pass

    def print_report(self):
        """打印评估报告"""
        # TODO: 实现报告打印
        pass
```

**答案**：

```python
class SimpleEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate(self, questions, retrieved_docs, relevant_docs):
        results = {
            "hit_rate": self.calculate_hit_rate(retrieved_docs, relevant_docs),
            "mrr": self.calculate_mrr(retrieved_docs, relevant_docs),
            "precision_at_1": self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=1),
            "precision_at_3": self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=3),
        }
        self.metrics = results
        return results

    def calculate_hit_rate(self, retrieved_docs, relevant_docs):
        hits = sum(
            1 for retrieved, relevant in zip(retrieved_docs, relevant_docs)
            if any(doc in relevant for doc in retrieved)
        )
        return hits / len(retrieved_docs)

    def calculate_mrr(self, retrieved_docs, relevant_docs):
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k):
        precisions = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            top_k = retrieved[:k]
            relevant_count = sum(1 for doc in top_k if doc in relevant)
            precisions.append(relevant_count / k)
        return sum(precisions) / len(precisions)

    def print_report(self):
        print("\n" + "="*50)
        print("评估报告")
        print("="*50)

        for metric, value in self.metrics.items():
            metric_name = metric.replace("_", " ").title()
            print(f"{metric_name:20s}: {value:.3f}")

        # 评级
        hit_rate = self.metrics.get("hit_rate", 0)
        if hit_rate > 0.85:
            rating = "优秀 ⭐⭐⭐⭐⭐"
        elif hit_rate > 0.7:
            rating = "良好 ⭐⭐⭐⭐"
        elif hit_rate > 0.5:
            rating = "中等 ⭐⭐⭐"
        else:
            rating = "需要改进 ⭐⭐"

        print(f"\n综合评级: {rating}")
        print("="*50)

# 使用示例
evaluator = SimpleEvaluator()
metrics = evaluator.evaluate(queries, retrieved_docs, relevant_docs)
evaluator.print_report()
```

---

## 第5章：综合项目

### 练习5.1：项目规划

**题目**：为一个"智能技术文档助手"项目制定计划

**要求**：
1. 定义核心功能（至少3个）
2. 选择技术栈
3. 确定评估指标
4. 估算开发时间

**答案示例**：

**项目名称**：TechDoc-AI

**核心功能**：
1. 文档上传和管理
2. 智能问答（基于文档）
3. 答案来源追踪

**技术栈**：
- 框架：LlamaIndex
- 向量库：Chroma
- LLM：GPT-3.5-turbo
- 嵌入：OpenAI text-embedding-3-small
- 界面：Streamlit

**评估指标**：
- Hit Rate > 0.6
- MRR > 0.5
- 响应时间 < 3秒

**开发时间**：
- Week 1：环境搭建 + 文档处理
- Week 2：RAG实现
- Week 3：界面开发
- Week 4：测试优化

---

### 练习5.2：代码实现

**题目**：实现项目的核心RAG引擎（参考第5章代码）

**提示**：
- 参考 `src/rag_engine.py`
- 实现文档加载、索引构建、查询功能

---

## 综合测试

### 测试1：概念理解（20分）

1. 解释RAG的核心价值（5分）
2. 列出RAG的5大组件（5分）
3. 比较RAG和Fine-tuning（10分）

### 测试2：代码实现（40分）

1. 实现文档加载函数（10分）
2. 实现文本分块函数（10分）
3. 实现向量检索函数（10分）
4. 实现完整的RAG查询（10分）

### 测试3：评估分析（20分）

1. 计算Hit Rate和MRR（10分）
2. 分析评估结果并提出优化建议（10分）

### 测试4：项目设计（20分）

1. 设计一个RAG应用场景（10分）
2. 制定完整的实现计划（10分）

---

## 答案自查

完成练习后，请使用以下清单检查：

- [ ] 理解了RAG的基本概念
- [ ] 能够实现基础的RAG系统
- [ ] 掌握了评估指标的计算
- [ ] 能够分析评估结果
- [ ] 完成了综合测试

**评分标准**：
- 90-100分：优秀 ⭐⭐⭐⭐⭐
- 80-89分：良好 ⭐⭐⭐⭐
- 70-79分：中等 ⭐⭐⭐
- 60-69分：及格 ⭐⭐
- <60分：需要加强学习

---

**祝你学习顺利！** 🎓
