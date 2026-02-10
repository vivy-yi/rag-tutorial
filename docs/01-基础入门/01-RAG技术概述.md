# 第1章：RAG技术概述

> 本章将带你全面了解RAG（Retrieval-Augmented Generation，检索增强生成）技术，掌握其核心概念、工作原理和应用场景，为后续深入学习打下坚实基础。

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解RAG的基本定义、核心价值和工作原理
- [ ] 清晰区分RAG、Fine-tuning和Prompt Engineering的适用场景
- [ ] 掌握RAG系统的五大核心组件及其作用
- [ ] 了解传统RAG到Agentic RAG的技术演进
- [ ] 根据实际需求选择合适的技术栈

**预计学习时间**：1.5小时
**难度等级**：⭐☆☆☆☆

---

## 前置知识

在开始本章学习前，你需要具备：

- [ ] **基础知识**：了解大语言模型（LLM）的基本概念
- [ ] **Python基础**：能够阅读和理解Python代码
- [ ] **机器学习概念**：理解向量、相似度等基本概念

**环境要求**：
- Python >= 3.9
- 网络连接（用于访问API和下载模型）
- 无需GPU（本章为理论介绍）

---

## 1.1 什么是RAG？

### 核心定义

**RAG（Retrieval-Augmented Generation，检索增强生成）**是一种结合了信息检索和文本生成的AI技术。它通过从外部知识库中检索相关文档，将这些文档作为上下文提供给大语言模型，从而生成更准确、更可靠的回答。

### 为什么需要RAG？

纯大语言模型存在以下问题：

#### 问题1：知识幻觉（Hallucination）
LLM可能会"自信地编造"错误信息。

**示例**：
```
用户：量子计算机的基本原理是什么？
纯LLM：量子计算机利用量子纠缠和量子叠加来同时处理大量数据...
       （可能包含过时或错误的技术细节）
```

**问题影响**：
- 医疗、法律等领域的错误信息可能造成严重后果
- 企业应用中不可靠的回答影响决策质量

#### 问题2：知识时效性限制
LLM的知识受训练数据截止日期限制。

**示例**：
```
用户：2024年最新的AI模型有哪些？
纯LLM：我的训练数据截止到2023年，无法回答...
```

#### 问题3：专业知识缺乏
通用模型难以覆盖特定领域的深度知识。

**示例**：
```
用户：我们公司的财务报销流程是什么？
纯LLM：我不知道贵公司的内部规定...
```

### RAG如何解决这些问题？

**核心思想**：让LLM"开卷考试"而非"闭卷考试"

```
传统LLM（闭卷考试）：
  用户问题 → LLM → 回答
             ↑
      仅依赖训练知识

RAG系统（开卷考试）：
  用户问题 → 检索相关文档 → LLM + 文档 → 回答
                ↑                    ↑
         外部知识库            结合文档生成
```

### RAG的核心价值

| 价值 | 说明 | 效果 |
|------|------|------|
| **准确性提升** | 基于真实文档生成答案 | 减少幻觉 |
| **知识更新** | 无需重新训练即可更新知识 | 实时性 |
| **可解释性** | 显示参考来源 | 可追溯 |
| **领域适配** | 轻松加入专业文档 | 专业化 |
| **成本效益** | 无需微调模型 | 低成本 |

### RAG vs 其他技术对比

#### 对比表格

| 技术 | 工作原理 | 知识更新 | 成本 | 适用场景 |
|------|---------|---------|------|---------|
| **Prompt Engineering** | 优化提示词 | 实时 | 极低 | 简单任务、快速原型 |
| **RAG** | 检索+生成 | 实时 | 中 | 需要准确知识、频繁更新 |
| **Fine-tuning** | 模型微调 | 需重新训练 | 高 | 特定风格、格式统一 |

#### 详细对比

**1. RAG vs Prompt Engineering**

```python
# Prompt Engineering示例
prompt = """
你是一个专业的客服人员。
请回答用户的问题：公司如何申请年假？
"""

# RAG示例
prompt = """
你是一个专业的客服人员。
基于以下公司政策文档回答用户问题：

【相关文档】
- 员工手册第3章：年假申请需提前3天提交...
- HR流程说明：通过OA系统填写申请表...

用户问题：公司如何申请年假？
"""
```

**区别**：
- **Prompt Engineering**：依赖LLM内置知识，适合通用任务
- **RAG**：提供具体文档，适合需要准确答案的任务

**2. RAG vs Fine-tuning**

```
Fine-tuning（知识内化）：
  训练数据 → 模型权重更新 → 新模型
  ↑
  耗时、昂贵、需重复训练

RAG（知识外挂）：
  知识库 → 检索 → 生成答案
  ↑
  实时、便宜、即插即用
```

**选择建议**：

- **使用RAG**：当知识经常变化、需要高准确性时
- **使用Fine-tuning**：当需要特定输出风格或格式时
- **组合使用**：先Fine-tuning学习格式，再RAG提供知识

### RAG的应用场景全景

#### 1. 企业知识管理

**场景**：企业内部文档问答

```
用户问题：公司的差旅补贴标准是什么？
RAG系统：
  1. 检索财务制度文档
  2. 定位"差旅补贴"章节
  3. 生成准确答案：
     "根据财务制度v3.0，一线城市住宿补贴800元/天..."
  4. 显示来源：财务制度.pdf 第12页
```

**价值**：
- 新员工快速获取信息
- 减少重复性咨询
- 知识资产数字化

#### 2. 客户服务

**场景**：产品使用咨询

```
用户问题：如何重置路由器？
RAG系统：
  1. 检索产品手册、FAQ
  2. 提取具体步骤
  3. 生成带步骤的答案
  4. 附上相关视频链接
```

**价值**：
- 24/7自动客服
- 降低人工成本
- 提升响应速度

#### 3. 教育培训

**场景**：课程材料问答

```
学生问题：Python中的装饰器是什么？
RAG系统：
  1. 检索课程讲义、代码示例
  2. 解释概念 + 提供代码
  3. 推荐相关练习
```

**价值**：
- 个性化学习助手
- 即时答疑
- 知识关联推荐

#### 4. 法律/医疗专业领域

**场景**：案例查询、医学诊断辅助

```
医生：症状为持续发热、乏力，可能是什么疾病？
RAG系统：
  1. 检索医学文献、临床指南
  2. 提供可能的疾病列表
  3. 标注来源和可信度
```

**注意**：
- 必须强调"辅助"而非"决策"
- 需要专业审核
- 高准确性要求

#### 5. 技术文档助手

**场景**：API文档查询

```
开发者：Pandas如何处理缺失值？
RAG系统：
  1. 检索官方文档
  2. 提供代码示例
  3. 说明参数含义
  4. 展示实际效果
```

**价值**：
- 提升开发效率
- 减少查文档时间
- 代码示例可直接使用

### 快速体验：一个简单的RAG系统

让我们通过一个简化示例，直观理解RAG的工作流程。

```python
# 文件名：01_simple_rag_demo.py
"""
最简RAG系统演示
仅用于理解概念，非生产代码
"""

# 步骤1：准备知识库（简化版，实际从文件加载）
knowledge_base = [
    {
        "id": 1,
        "content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。"
    },
    {
        "id": 2,
        "content": "JavaScript主要用于Web开发，可以在浏览器中运行。"
    },
    {
        "id": 3,
        "content": "Rust是一种系统编程语言，注重内存安全和性能。"
    }
]

# 步骤2：简单的检索函数（基于关键词匹配）
def retrieve_documents(query, kb, top_k=2):
    """
    检索相关文档

    Args:
        query: 用户问题
        kb: 知识库
        top_k: 返回前K个结果

    Returns:
        相关文档列表
    """
    # 简单的关键词匹配（实际应该使用向量相似度）
    query_lower = query.lower()

    # 计算每个文档的相关性分数
    scores = []
    for doc in kb:
        content_lower = doc["content"].lower()
        # 简单计算：统计问题中出现在文档中的词数
        score = sum(1 for word in query_lower.split() if word in content_lower)
        scores.append((doc, score))

    # 按分数排序，返回top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scores[:top_k] if score > 0]

# 步骤3：构建提示词
def build_prompt(query, retrieved_docs):
    """
    构建包含检索文档的提示词

    Args:
        query: 用户问题
        retrieved_docs: 检索到的文档

    Returns:
        完整提示词
    """
    context = "\n".join([
        f"文档{i+1}: {doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = f"""
基于以下文档回答用户问题。如果文档中没有相关信息，请明确说明。

【参考文档】
{context}

【用户问题】
{query}

【回答】
"""
    return prompt

# 步骤4：模拟LLM生成（实际应调用真实LLM）
def generate_response(prompt):
    """
    生成回答

    实际应用中，这里应该调用OpenAI API或其他LLM
    这里简化为返回提示词本身
    """
    print("=== 发送给LLM的提示词 ===")
    print(prompt)
    print("\n=== LLM会基于上述信息生成回答 ===\n")
    return "[这里应该是LLM生成的回答]"

# 步骤5：完整的RAG流程
def simple_rag_pipeline(query):
    """
    完整的RAG流程

    Args:
        query: 用户问题

    Returns:
        回答和参考文档
    """
    print(f"用户问题: {query}\n")

    # 1. 检索相关文档
    print("步骤1: 检索相关文档")
    retrieved_docs = retrieve_documents(query, knowledge_base)
    print(f"检索到 {len(retrieved_docs)} 个相关文档\n")

    if not retrieved_docs:
        return "抱歉，知识库中没有找到相关信息。", []

    # 2. 构建提示词
    print("步骤2: 构建提示词")
    prompt = build_prompt(query, retrieved_docs)

    # 3. 生成回答
    print("步骤3: 生成回答\n")
    answer = generate_response(prompt)

    # 4. 返回结果和来源
    return answer, retrieved_docs

# 运行示例
if __name__ == "__main__":
    # 测试问题
    test_queries = [
        "Python是什么时候创建的？",
        "Rust语言的特点是什么？",
        "如何学习Go语言？"  # 知识库中没有的信息
    ]

    print("=" * 60)
    print("简单RAG系统演示")
    print("=" * 60)
    print()

    for query in test_queries:
        print("\n" + "=" * 60)
        answer, sources = simple_rag_pipeline(query)

        if sources:
            print("=== 最终答案 ===")
            print(answer)
            print("\n=== 参考来源 ===")
            for doc in sources:
                print(f"- 文档ID: {doc['id']}")
                print(f"  内容: {doc['content']}")
```

#### 运行示例

```bash
# 运行演示代码
python 01_simple_rag_demo.py
```

#### 预期输出

```
============================================================
简单RAG系统演示
============================================================

用户问题: Python是什么时候创建的？

步骤1: 检索相关文档
检索到 1 个相关文档

步骤2: 构建提示词
步骤3: 生成回答

=== 发送给LLM的提示词 ===
基于以下文档回答用户问题。如果文档中没有相关信息，请明确说明。

【参考文档】
文档1: Python是一种高级编程语言，由Guido van Rossum于1991年创建。

【用户问题】
Python是什么时候创建的？

【回答】

=== LLM会基于上述信息生成回答 ===

=== 最终答案 ===
[这里应该是LLM生成的回答]

=== 参考来源 ===
- 文档ID: 1
  内容: Python是一种高级编程语言，由Guido van Rossum于1991年创建。
```

#### 代码解析

**关键要点**：

1. **知识库**（knowledge_base）
   - 实际应用中是文档、PDF、网页等
   - 需要预先处理和索引

2. **检索函数**（retrieve_documents）
   - 这里使用简单的关键词匹配
   - 实际应该使用向量相似度计算

3. **提示词构建**（build_prompt）
   - 将检索到的文档注入提示词
   - 这是RAG的核心思想

4. **LLM生成**（generate_response）
   - 实际应该调用OpenAI API
   - 基于提供的文档生成答案

**进一步优化方向**：
- 使用真实的嵌入模型和向量数据库
- 调用真实的LLM API
- 添加文档来源追踪
- 实现更智能的检索策略

---

## 1.2 RAG的核心组件

一个完整的RAG系统由5个核心组件构成，每个组件都至关重要。

### 组件总览

```
┌─────────────────────────────────────────────────────────┐
│                    RAG系统架构                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣ 文档加载器              │
│     └─ 支持PDF、Word、网页、数据库等                      │
│                                                         │
│  2️⃣ 文本分块器                  │
│     └─ 将长文档切分成可管理的小块                        │
│                                                         │
│  3️⃣ 嵌入模型           │
│     └─ 将文本转换为向量表示                             │
│                                                         │
│  4️⃣ 向量数据库                │
│     └─ 存储和检索向量                                   │
│                                                         │
│  5️⃣ 大语言模型                    │
│     └─ 基于检索到的文档生成答案                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 组件1：文档加载器（Document Loaders）

#### 作用
从各种数据源加载原始文档内容。

#### 支持的数据源

| 数据源类型 | 示例 | 常用加载器 |
|-----------|------|-----------|
| **本地文件** | PDF, Word, TXT | PyPDFLoader, Docx2txtLoader |
| **网页内容** | HTML, Markdown | WebBaseLoader, UnstructuredURLLoader |
| **数据库** | MySQL, PostgreSQL | SQLDatabaseLoader |
| **云存储** | S3, Google Drive | S3FileLoader, GDriveLoader |
| **Notion** | Notion页面 | NotionLoader |
| **代码仓库** | GitHub | GitHubLoader |

#### 代码示例

```python
# 示例：加载多种类型的文档
from llama_index import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

# 1. 加载本地目录中的所有PDF
def load_local_documents(directory_path):
    """
    从本地目录加载文档

    Args:
        directory_path: 文档目录路径

    Returns:
        文档列表
    """
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".pdf", ".txt", ".md"],  # 只加载这些格式
        recursive=True  # 递归加载子目录
    )
    documents = reader.load_data()
    return documents

# 2. 加载网页内容
def load_web_pages(urls):
    """
    从URL加载网页内容

    Args:
        urls: URL列表

    Returns:
        文档列表
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    return documents

# 3. 加载单个PDF文件
def load_single_pdf(file_path):
    """
    加载单个PDF文件

    Args:
        file_path: PDF文件路径

    Returns:
        文档对象
    """
    from llama_index.readers.file import PyPDFReader

    loader = PyPDFReader()
    documents = loader.load_data(file_path)
    return documents

# 使用示例
if __name__ == "__main__":
    # 加载本地文档
    docs = load_local_documents("./data/documents")
    print(f"加载了 {len(docs)} 个文档")

    # 加载网页
    urls = ["https://example.com/article1", "https://example.com/article2"]
    web_docs = load_web_pages(urls)
    print(f"从网页加载了 {len(web_docs)} 个文档")
```

#### 最佳实践

1. **批量处理**：一次加载多个文档，避免频繁I/O
2. **错误处理**：处理损坏文件和编码问题
3. **元数据保留**：保存文件名、路径、修改时间等信息
4. **进度监控**：对于大量文件，显示加载进度

### 组件2：文本分块器（Text Splitters）

#### 为什么需要分块？

**问题**：文档太长，无法一次处理

```
PDF文档（100页） → LLM无法处理
    ↓
分块（每块500字） → 可处理
```

#### 分块策略对比

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **固定长度** | 按字符/词数切分 | 简单快速 | 可能切断语义 | 通用场景 |
| **递归分块** | 多级分隔符 | 保持段落完整 | 仍可能切断句子 | 结构化文档 |
| **语义分块** | 基于语义边界 | 语义完整 | 计算成本高 | 长文本、重要文档 |
| **专用分块** | 按代码/公式分块 | 保持结构完整 | 需要专门实现 | 代码文档 |

#### 代码示例

```python
# 示例：不同的分块策略
from llama_index.text_splitter import (
    SentenceSplitter,
    TokenTextSplitter,
)

# 1. 固定长度分块（按字符）
def chunk_by_char(text, chunk_size=500, chunk_overlap=50):
    """
    按字符数分块

    Args:
        text: 输入文本
        chunk_size: 每块大小（字符数）
        chunk_overlap: 块之间重叠大小

    Returns:
        分块列表
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"  # 优先按段落分割
    )
    chunks = splitter.split_text(text)
    return chunks

# 2. 按Token分块
def chunk_by_token(text, chunk_size=1000, chunk_overlap=100):
    """
    按Token数分块（更符合LLM处理方式）

    Args:
        text: 输入文本
        chunk_size: 每块大小（Token数）
        chunk_overlap: 块之间重叠大小

    Returns:
        分块列表
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base"  # OpenAI的编码方式
    )
    chunks = splitter.split_text(text)
    return chunks

# 3. 递归分块（推荐）
def chunk_recursive(documents, chunk_size=1000, chunk_overlap=200):
    """
    递归分块：尝试多种分隔符

    分隔符优先级：段落 > 句子 > 词 > 字符

    Args:
        documents: 文档列表
        chunk_size: 每块大小
        chunk_overlap: 重叠大小

    Returns:
        分块列表
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

# 使用示例
if __name__ == "__main__":
    sample_text = """
    人工智能（AI）是计算机科学的一个分支。
    它致力于创建能够执行通常需要人类智能的任务的系统。

    机器学习是AI的一个子集。
    它使计算机能够从数据中学习，而不是被明确编程。

    深度学习是机器学习的一种方法。
    它使用多层神经网络来模拟人脑的工作方式。
    """

    # 固定长度分块
    chunks_char = chunk_by_char(sample_text, chunk_size=50)
    print(f"字符分块: {len(chunks_char)} 个块")

    # Token分块
    chunks_token = chunk_by_token(sample_text, chunk_size=20)
    print(f"Token分块: {len(chunks_token)} 个块")
```

#### 分块参数选择指南

**chunk_size（块大小）**：
- **太小（<200）**：信息不足，上下文缺失
- **推荐（500-1500）**：平衡信息和性能
- **太大（>2000）**：检索不精确，噪音多

**chunk_overlap（重叠大小）**：
- **作用**：保持上下文连续性
- **推荐值**：chunk_size的10-20%
- **示例**：chunk_size=1000，overlap=100-200

**分块可视化**：

```
原文: [A B C D E F G H I J]

chunk_size=3, overlap=1:

块1: [A B C]
块2: [C D E]  ← 'C'重叠
块3: [E F G]  ← 'E'重叠
块4: [G H I]  ← 'G'重叠
块5: [I J]
```

### 组件3：嵌入模型（Embedding Models）

#### 什么是嵌入？

**嵌入（Embedding）**：将文本转换为数值向量的过程

```
文本："人工智能很强大"
    ↓
嵌入模型
    ↓
向量：[0.23, -0.45, 0.67, ..., 0.12]  (768维)
```

#### 为什么使用嵌入？

**核心思想**：语义相似的文本，在向量空间中距离更近

```
向量空间可视化（2D投影）：

         猫
          🐱
         /|\
        / | \
    狗  |  |  老虎
   🐶────┼──┼──🐯
        |  |
      电脑 |
       💻

"猫" 和 "狗" 距离近 → 语义相似
"电脑" 和 "猫" 距离远 → 语义不同
```

#### 主流嵌入模型对比

| 模型 | 维度 | 特点 | 适用场景 | 成本 |
|------|------|------|---------|------|
| **OpenAI text-embedding-3** | 1536/3072 | 性能优秀，稳定 | 通用，推荐生产使用 | 按API调用收费 |
| **BGE系列** | 768/1024 | 中文优化，开源 | 中文应用，私有部署 | 免费 |
| **E5系列** | 768/1024 | 多语言支持 | 国际化应用 | 免费 |
| **Sentence-Transformers** | 384-768 | 轻量快速 | 资源受限环境 | 免费 |

#### 代码示例

```python
# 示例：使用不同的嵌入模型
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. OpenAI嵌入（推荐生产环境）
def embed_with_openai(texts, model="text-embedding-3-small"):
    """
    使用OpenAI嵌入模型

    Args:
        texts: 文本列表
        model: 模型名称

    Returns:
        嵌入向量列表
    """
    from openai import OpenAI

    client = OpenAI()  # 需要设置API key

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        embeddings.append(response.data[0].embedding)

    return embeddings

# 2. 开源模型（推荐私有部署）
def embed_with_sentence_transformer(texts, model_name="BAAI/bge-small-zh-v1.5"):
    """
    使用Sentence Transformers嵌入模型

    Args:
        texts: 文本列表
        model_name: 模型名称

    Returns:
        嵌入向量列表
    """
    from sentence_transformers import SentenceTransformer

    # 加载模型（首次会下载）
    model = SentenceTransformer(model_name)

    # 生成嵌入
    embeddings = model.encode(texts)

    return embeddings

# 3. 计算相似度
def compute_similarity(query, documents, embeddings):
    """
    计算查询和文档的相似度

    Args:
        query: 查询文本
        documents: 文档列表
        embeddings: 文档的嵌入向量

    Returns:
        相似度分数列表
    """
    # 生成查询的嵌入
    query_embedding = embed_with_sentence_transformer([query])[0]

    # 计算余弦相似度
    similarities = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    # 返回排序后的结果
    results = list(zip(documents, similarities))
    results.sort(key=lambda x: x[1], reverse=True)

    return results

# 使用示例
if __name__ == "__main__":
    # 示例文档
    docs = [
        "Python是一种编程语言",
        "Java也是一种编程语言",
        "今天天气很好",
        "我喜欢吃苹果"
    ]

    print("生成嵌入向量...")
    embeddings = embed_with_sentence_transformer(docs)

    # 查询相似文档
    query = "编程语言有哪些？"
    results = compute_similarity(query, docs, np.array(embeddings))

    print(f"\n查询: {query}\n")
    for doc, score in results[:3]:
        print(f"相似度 {score:.3f}: {doc}")
```

### 组件4：向量数据库（Vector Stores）

#### 为什么需要向量数据库？

**问题**：如何高效检索百万级向量？

```
朴素方法：
  对于每个查询，遍历所有向量计算相似度
  → 时间复杂度O(n)，太慢！

向量数据库：
  使用索引算法（如HNSW）
  → 时间复杂度O(log n)，快很多！
```

#### 主流向量数据库对比

| 数据库 | 特点 | 难度 | 适用场景 |
|--------|------|------|---------|
| **Chroma** | 轻量、易用 | ⭐☆☆☆☆ | 原型开发、学习 |
| **Qdrant** | 高性能、易部署 | ⭐⭐☆☆☆ | 中小型生产 |
| **Milvus** | 功能全面、可扩展 | ⭐⭐⭐☆☆ | 大规模生产 |
| **Pinecone** | 云托管服务 | ⭐☆☆☆☆ | 快速上线（付费）|
| **Weaviate** | 模块化、GraphQL | ⭐⭐⭐☆☆ | 复杂查询场景 |

#### 代码示例（Chroma）

```python
# 示例：使用Chroma向量数据库
import chromadb
from chromadb.config import Settings

# 1. 初始化Chroma
def init_chroma(persist_directory="./chroma_db"):
    """
    初始化Chroma客户端

    Args:
        persist_directory: 数据持久化目录

    Returns:
        Chroma客户端
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client

# 2. 创建集合
def create_collection(client, name="my_documents"):
    """
    创建向量集合

    Args:
        client: Chroma客户端
        name: 集合名称

    Returns:
        集合对象
    """
    # 如果集合已存在，获取它；否则创建新的
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    return collection

# 3. 添加文档
def add_documents(collection, documents, embeddings, metadatas=None):
    """
    添加文档到集合

    Args:
        collection: 集合对象
        documents: 文本列表
        embeddings: 嵌入向量列表
        metadatas: 元数据列表（可选）

    Returns:
        添加的文档数量
    """
    # 生成唯一ID
    ids = [f"doc_{i}" for i in range(len(documents))]

    # 添加到集合
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)

# 4. 查询相似文档
def query_similar(collection, query_embedding, n_results=3):
    """
    查询最相似的文档

    Args:
        collection: 集合对象
        query_embedding: 查询向量
        n_results: 返回结果数量

    Returns:
        查询结果
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results

# 5. 完整示例
def vector_db_example():
    """
    完整的向量数据库使用示例
    """
    # 初始化
    client = init_chroma()
    collection = create_collection(client, "demo_docs")

    # 准备数据
    documents = [
        "Python是一种高级编程语言",
        "JavaScript主要用于Web开发",
        "Rust注重内存安全和性能",
        "Go语言适合并发编程"
    ]

    # 生成嵌入（使用前面定义的函数）
    embeddings = embed_with_sentence_transformer(documents)

    # 添加到数据库
    add_documents(collection, documents, embeddings)
    print(f"添加了 {len(documents)} 个文档\n")

    # 查询
    query = "什么语言适合系统编程？"
    query_embedding = embed_with_sentence_transformer([query])[0]

    results = query_similar(collection, query_embedding, n_results=2)

    print(f"查询: {query}\n")
    print("最相关的文档:")
    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    ), 1):
        print(f"{i}. {doc}")
        print(f"   相似度: {1 - distance:.3f}\n")

# 运行示例
if __name__ == "__main__":
    vector_db_example()
```

### 组件5：大语言模型（LLMs）

#### 作用

基于检索到的文档生成自然语言回答。

#### 主流LLM选择

| 模型 | 提供商 | 特点 | 成本 | 适用场景 |
|------|--------|------|------|---------|
| **GPT-4** | OpenAI | 性能最强 | 高 | 复杂任务、生产环境 |
| **GPT-3.5** | OpenAI | 快速便宜 | 中 | 通用场景、高并发 |
| **Claude** | Anthropic | 长文本优秀 | 中 | 长文档处理 |
| **Llama系列** | Meta | 开源可部署 | 免费 | 私有部署、定制需求 |
| **Qwen** | 阿里 | 中文优化 | 免费 | 中文应用 |

#### 代码示例

```python
# 示例：使用LLM生成答案
from openai import OpenAI

def generate_answer_with_llm(query, context_documents, model="gpt-3.5-turbo"):
    """
    使用LLM生成答案

    Args:
        query: 用户问题
        context_documents: 检索到的相关文档
        model: LLM模型名称

    Returns:
        生成的答案
    """
    client = OpenAI()  # 需要设置API key

    # 构建上下文
    context = "\n\n".join([
        f"【文档{i+1}}\n{doc}"
        for i, doc in enumerate(context_documents)
    ])

    # 构建提示词
    prompt = f"""
你是一个专业的助手。请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{query}

要求：
1. 基于文档内容回答
2. 如果文档中没有相关信息，明确说明
3. 回答要准确、简洁

回答：
"""

    # 调用LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的问答助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # 控制随机性
        max_tokens=500    # 限制回答长度
    )

    answer = response.choices[0].message.content
    return answer

# 使用示例
if __name__ == "__main__":
    query = "Python有什么特点？"

    context_docs = [
        "Python是一种高级编程语言，由Guido van Rossum创建。",
        "Python的特点是语法简洁、易学易用、应用广泛。",
        "Python可用于Web开发、数据分析、人工智能等领域。"
    ]

    answer = generate_answer_with_llm(query, context_docs)

    print(f"问题: {query}\n")
    print(f"回答: {answer}")
```

### 完整RAG流程图

```
用户问题
    ↓
┌─────────────────────────────────────────────┐
│ 1. 文档加载器                               │
│    从PDF、网页等加载原始文档                │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 2. 文本分块器                               │
│    将长文档切分成小块（chunk）              │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 3. 嵌入模型                                 │
│    将文本块转换为向量（embedding）          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 4. 向量数据库                               │
│    存储向量，快速检索相似文档               │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 5. 大语言模型（LLM）                        │
│    基于检索到的文档生成答案                 │
└─────────────────┬───────────────────────────┘
                  ↓
              最终答案
```

---

## 1.3 传统RAG vs Agentic RAG

### RAG技术演进

RAG技术经历了三个阶段的演进：

```
阶段1: Naive RAG（传统RAG）
  ↓
阶段2: Advanced RAG（高级RAG）
  ↓
阶段3: Agentic RAG（智能体RAG）
```

### Naive RAG（传统RAG）

#### 工作流程

```
用户问题
    ↓
[检索] → 向量数据库 → 相关文档
    ↓
[生成] → LLM → 答案
```

#### 优点
- ✅ 实现简单
- ✅ 易于理解
- ✅ 适合简单问题

#### 局限性

**1. 检索质量依赖查询表达**

```
用户问题： "那个...就是修电脑的东西"
实际意图： "维修工具"

问题：检索词不准确 → 检索效果差
```

**2. 无法处理复杂问题**

```
用户问题： "比较Python和JavaScript在Web开发中的优劣"
问题：需要多步推理 → 传统RAG难以处理
```

**3. 检索结果可能不相关**

```
用户问题： "如何使用Pandas处理CSV文件？"
检索到： ["Python是什么", "CSV文件格式", "数据处理简介"]

问题：检索质量不稳定 → 答案质量差
```

### Advanced RAG（高级RAG）

#### 改进点

在检索前、检索中、检索后增加优化步骤。

```
用户问题
    ↓
[检索前优化]
  - 查询重写
  - 查询扩展
  - HyDE（假设文档嵌入）
    ↓
[检索]
  - 混合检索（向量+关键词）
  - 递归检索
    ↓
[检索后优化]
  - 重排序（Reranking）
  - 过滤
    ↓
[生成] → 答案
```

#### 关键技术

| 技术 | 作用 | 效果提升 |
|------|------|---------|
| **查询重写** | 优化查询表达 | +15-20% |
| **混合检索** | 结合向量和关键词 | +10-15% |
| **重排序** | 精细化排序 | +20-30% |

**将在模块2详细讲解这些技术！**

### Agentic RAG（智能体RAG）

#### 核心思想

让RAG系统具备"智能"，能够：
- 🔍 **自我反思**：判断答案是否满意
- 🔄 **动态调整**：根据反馈调整策略
- 🛠️ **工具使用**：主动调用多个工具
- 📋 **任务规划**：将复杂问题分解

#### 工作流程

```
用户问题
    ↓
[智能体分析]
  - 问题理解
  - 任务分解
    ↓
[动态检索策略]
  - 判断需要什么信息
  - 选择合适的检索方法
    ↓
[迭代优化]
  - 评估答案质量
  - 不满意则重新检索
    ↓
[多步骤推理]
  - 整合多次检索结果
  - 进行逻辑推理
    ↓
最终答案
```

#### 对比示例

**场景**：用户询问复杂的多步问题

```
问题： "分析特斯拉公司2023年的财务状况，并评估其投资价值"

Naive RAG：
  1. 检索"特斯拉 财务"
  2. 直接生成答案
  → 问题：可能不全面，缺少分析

Agentic RAG：
  1. 分析：这是个复杂问题，需要多步处理
  2. 规划：
     - 步骤1：检索2023财报数据
     - 步骤2：检索关键财务指标
     - 步骤3：检索行业对比数据
     - 步骤4：检索投资分析框架
  3. 执行：依次检索每个子问题
  4. 整合：综合所有信息，生成分析报告
  5. 反思：检查答案是否完整，必要时补充检索
  → 结果：全面、深入、结构化的答案
```

### 三种方案详细对比

| 维度 | Naive RAG | Advanced RAG | Agentic RAG |
|------|-----------|--------------|-------------|
| **实现复杂度** | ⭐☆☆☆☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ |
| **检索质量** | ⭐⭐☆☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |
| **推理能力** | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | ⭐⭐⭐⭐⭐ |
| **开发成本** | 低 | 中 | 高 |
| **运行成本** | 低 | 中 | 高（多轮LLM调用）|
| **适用场景** | 简单问答 | 中等复杂度 | 复杂推理任务 |
| **响应时间** | 快（1-2秒）| 中（2-5秒）| 慢（5-10秒+）|

### 智能体文档工作流（ADW）

Agentic RAG的一个典型应用是**智能体文档工作流（Agent Document Workflow, ADW）**。

#### 传统文档处理 vs ADW

```
传统流程：
  文档 → 人工阅读 → 提取信息 → 人工分析 → 报告
  ↑
  耗时、易错、难扩展

ADW流程：
  文档 → 智能体自动处理 → 结构化信息 → 自动分析 → 报告
  ↑
  快速、准确、可扩展
```

#### ADW工作流程

```
1. [文档理解]
   - 识别文档类型（合同/报告/发票等）
   - 提取关键信息

2. [信息提取]
   - 识别实体（人名/公司/日期等）
   - 提取关系

3. [质量检查]
   - 验证信息完整性
   - 标记不确定部分

4. [内容生成]
   - 生成摘要
   - 生成报告

5. [人工审核]
   - 展示结果
   - 接受反馈
   - 必要时修正
```

**将在模块3深入讲解Agentic RAG！**

---

## 1.4 RAG技术栈选择

### 主流框架对比

#### LlamaIndex vs LangChain

| 特性 | LlamaIndex | LangChain |
|------|------------|-----------|
| **设计理念** | 专注RAG和数据应用 | 通用LLM应用框架 |
| **学习曲线** | 较平缓 | 相对陡峭 |
| **RAG功能** | 开箱即用，优化好 | 需要自己组合 |
| **数据连接器** | 丰富（150+） | 较少（50+） |
| **灵活性** | 中等 | 高 |
| **文档质量** | 优秀 | 良好 |
| **社区活跃度** | 高 | 很高 |
| **推荐场景** | RAG应用 | 通用LLM应用 |

#### 选择建议

**选择LlamaIndex**：
- ✅ 主要做RAG应用
- ✅ 希望快速上手
- ✅ 需要丰富的数据连接器
- ✅ 重视代码简洁性

**选择LangChain**：
- ✅ 需要构建复杂的LLM应用
- ✅ 需要高度定制化
- ✅ 有更多编程经验
- ✅ 需要社区广泛支持

**本教程选择**：**LlamaIndex**
- 更适合RAG学习
- API更简洁
- 专注于数据应用

### 向量数据库选择

#### 决策树

```
开始
  ↓
需要快速原型/学习？
  ├─ 是 → Chroma（最简单）
  └─ 否 ↓
    需要云托管（不想自己运维）？
      ├─ 是 → Pinecone（付费）或 Zilliz Cloud
      └─ 否 ↓
        数据量（百万级向量）？
          ├─ < 100万 → Qdrant
          └─ > 100万 ↓
            需要分布式/高可用？
              ├─ 是 → Milvus
              └─ 否 → Qdrant
```

#### 详细对比

| 数据库 | 部署难度 | 性能 | 扩展性 | 成本 | 推荐场景 |
|--------|---------|------|--------|------|---------|
| **Chroma** | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | ⭐☆☆☆☆ | 免费 | 学习、原型 |
| **Qdrant** | ⭐⭐☆☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | 免费 | 中小型生产 |
| **Milvus** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 免费 | 大规模生产 |
| **Pinecone** | ⭐☆☆☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | 付费 | 快速上线 |

**本教程选择**：
- **学习阶段**：Chroma（简单）
- **生产阶段**：Milvus（模块4）

### LLM选择指南

#### 决策因素

```
1. 语言要求
   ├─ 中文为主 → Qwen、ChatGLM
   ├─ 英文为主 → GPT系列、Llama
   └─ 多语言 → GPT-4、Claude

2. 部署方式
   ├─ 云API（方便） → GPT-4、Claude、Qwen API
   └─ 私有部署（安全） → Llama、Qwen、ChatGLM

3. 成本预算
   ├─ 有预算 → GPT-4（性能最佳）
   └─ 预算有限 → GPT-3.5或开源模型

4. 性能要求
   ├─ 高质量 → GPT-4、Claude
   └─ 一般 → GPT-3.5、开源模型
```

#### 推荐配置

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| **学习/开发** | GPT-3.5-turbo | 便宜、快速、够用 |
| **中文生产** | Qwen-72B | 中文好、可私有部署 |
| **高精度** | GPT-4 | 质量最高 |
| **成本敏感** | Llama-3-8B | 免费运行 |

### 完整技术栈推荐

#### 方案1：快速学习（推荐初学者）

```
框架：LlamaIndex
向量库：Chroma
LLM：GPT-3.5-turbo
嵌入：OpenAI text-embedding-3-small

优势：简单、快速、成本低
```

#### 方案2：私有部署（企业应用）

```
框架：LlamaIndex
向量库：Qdrant
LLM：Qwen-72B（自托管）
嵌入：BGE-large-zh（开源）

优势：数据安全、无API费用、可控
```

#### 方案3：大规模生产

```
框架：LlamaIndex
向量库：Milvus（分布式）
LLM：GPT-4（高质量）+ vLLM（加速）
嵌入：OpenAI text-embedding-3-large

优势：高性能、高质量、可扩展
```

### 技术栈选择决策表

回答以下问题，找到适合你的技术栈：

**Q1：你的主要目的是什么？**
- A. 学习RAG → 方案1
- B. 构建企业应用 → Q2

**Q2：数据是否敏感？**
- A. 是（不能上传到云）→ 方案2
- B. 否 → Q3

**Q3：数据量有多大？**
- A. < 10万文档 → Qdrant
- B. > 10万文档 → Milvus

**Q4：预算如何？**
- A. 有预算 → GPT-4
- B. 预算有限 → GPT-3.5或开源模型

---

## 总结

### 本章要点回顾

1. **RAG是什么**
   - 检索增强生成，结合检索和生成
   - 解决LLM的幻觉、时效性、专业知识问题

2. **RAG的5大核心组件**
   - 文档加载器：从各种数据源加载文档
   - 文本分块器：将长文档切分成可管理的块
   - 嵌入模型：将文本转换为向量
   - 向量数据库：高效存储和检索向量
   - 大语言模型：基于文档生成答案

3. **RAG的演进**
   - Naive RAG：简单直接的检索+生成
   - Advanced RAG：优化检索和结果
   - Agentic RAG：具备智能和推理能力

4. **技术栈选择**
   - 框架：LlamaIndex（适合RAG）
   - 向量库：学习用Chroma，生产用Milvus
   - LLM：学习用GPT-3.5，生产按需求选择

### 学习检查清单

- [ ] 理解RAG的基本概念和价值
- [ ] 能够描述RAG的完整工作流程
- [ ] 掌握5大核心组件的作用
- [ ] 了解Naive、Advanced、Agentic RAG的区别
- [ ] 能够根据需求选择合适的技术栈
- [ ] 运行了示例代码，体验了简单RAG

### 下一步学习

- **下一章**：[第2章：环境搭建与工具准备](./02-环境搭建与工具准备.md)
  - 安装Python和开发环境
  - 配置LlamaIndex和相关库
  - 准备示例数据集

- **相关章节**：
  - 第3章：基础RAG实现（动手编写第一个RAG系统）

### 扩展资源

#### 推荐阅读

1. **RAG原论文**
   - Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - 链接：[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
   - 核心观点：首次提出RAG概念

2. **LlamaIndex官方文档**
   - 链接：[https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
   - 特点：高质量的教程和示例

3. **LangChain RAG教程**
   - 链接：[https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)
   - 特点：详细的实现指南

#### 相关项目

- **LlamaIndex**: [https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index) - 数据框架
- **Chroma**: [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma) - 向量数据库
- **LangChain**: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) - LLM框架

---

## 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| **RAG** | Retrieval-Augmented Generation | 检索增强生成，结合检索和生成的AI技术 |
| **嵌入** | Embedding | 将文本转换为数值向量的过程 |
| **向量数据库** | Vector Database | 专门用于存储和检索向量的数据库 |
| **分块** | Chunking | 将长文档切分成小块的过程 |
| **幻觉** | Hallucination | LLM生成虚假或错误信息的现象 |
| **相似度** | Similarity | 衡量两个向量相关性的指标 |
| **LLM** | Large Language Model | 大语言模型 |

---

**返回目录** | **上一章**（无）| **下一章：环境搭建与工具准备**

---

**本章结束**

> 有任何问题或建议？欢迎提交Issue或PR到教程仓库！

---

**作者笔记**

> 本章是整个教程的基础，建议：
> 1. 仔细理解RAG的各个组件
> 2. 运行示例代码，获得直观感受
> 3. 思考你的应用场景，规划技术栈
> 4. 不要急于进入下一章，确保理解了基础概念
>
> RAG的核心思想很简单，但实现细节有很多优化空间。
> 本章建立了基础概念，后续章节会逐步深入各种优化技术。
