# 第3章：基础RAG实现

> 理论结合实践！本章将带你从零开始构建第一个完整的RAG系统，每一步都有详细说明和完整代码。

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 使用LlamaIndex加载各种格式的文档
- [ ] 实现多种文本分块策略
- [ ] 理解并使用嵌入模型
- [ ] 构建向量检索系统
- [ ] 实现完整的RAG问答流程
- [ ] 评估RAG系统的性能

**预计学习时间**：2.5小时
**难度等级**：⭐⭐☆☆☆

---

## 前置知识

在开始本章学习前，你需要：

- [ ] 完成第2章的环境搭建
- [ ] 已安装LlamaIndex和Chroma
- [ ] 准备好示例数据集
- [ ] 配置好OpenAI API密钥

**环境要求**：
- Python 3.10+
- 已安装llama-index-core
- 已安装chromadb
- 已设置OPENAI_API_KEY环境变量

---

## 3.1 文档加载与处理

### 文档加载器详解

#### 为什么需要文档加载器？

文档数据以各种格式存在（PDF、Word、网页等），需要统一转换为RAG系统可处理的格式。

```
原始数据（多种格式）
    ↓
文档加载器
    ↓
统一格式（Document对象）
    ↓
RAG系统处理
```

### LlamaIndex文档加载器

#### 基础用法

```python
# 文件名：03_01_document_loading.py
"""
文档加载示例
演示如何加载各种格式的文档
"""

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.file import PyPDFReader
from pathlib import Path

# 1. 加载整个目录
def load_directory(directory_path: str):
    """
    从目录加载所有文档

    Args:
        directory_path: 目录路径

    Returns:
        文档列表
    """
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".txt", ".md", ".pdf"],  # 只加载这些格式
        recursive=True  # 递归加载子目录
    )

    documents = reader.load_data()
    return documents

# 2. 加载单个PDF文件
def load_single_pdf(file_path: str):
    """
    加载单个PDF文件

    Args:
        file_path: PDF文件路径

    Returns:
        文档列表
    """
    loader = PyPDFReader()
    documents = loader.load_data(file_path)
    return documents

# 3. 加载多个文件
def load_specific_files(file_paths: list):
    """
    加载指定的文件列表

    Args:
        file_paths: 文件路径列表

    Returns:
        文档列表
    """
    reader = SimpleDirectoryReader(
        input_files=file_paths
    )
    documents = reader.load_data()
    return documents

# 4. 查看文档信息
def inspect_documents(documents):
    """
    查看文档的详细信息

    Args:
        documents: 文档列表
    """
    print(f"总文档数: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        print(f"文档 {i}:")
        print(f"  - 元数据: {doc.metadata}")
        print(f"  - 字符数: {len(doc.text)}")
        print(f"  - 预览: {doc.text[:100]}...")
        print()

# 使用示例
if __name__ == "__main__":
    # 准备测试数据
    import os
    os.makedirs("data/test_docs", exist_ok=True)

    # 创建测试文件
    test_files = {
        "data/test_docs/doc1.txt": "这是第一个测试文档。\n包含两行内容。",
        "data/test_docs/doc2.txt": "这是第二个测试文档。\n用于测试批量加载功能。"
    }

    for filepath, content in test_files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"创建测试文件: {filepath}")

    print("\n" + "="*60)
    print("测试1: 加载目录")
    print("="*60 + "\n")

    # 测试1：加载目录
    docs = load_directory("data/test_docs")
    inspect_documents(docs)

    print("="*60)
    print("测试2: 加载指定文件")
    print("="*60 + "\n")

    # 测试2：加载指定文件
    specific_docs = load_specific_files([
        "data/test_docs/doc1.txt",
        "data/test_docs/doc2.txt"
    ])
    inspect_documents(specific_docs)

    print("\n✓ 文档加载测试完成！")
```

### 高级加载功能

#### 自定义元数据

```python
# 为文档添加自定义元数据
from llama_index.core import Document

def load_with_metadata(file_path: str, metadata: dict):
    """
    加载文档并添加自定义元数据

    Args:
        file_path: 文件路径
        metadata: 元数据字典

    Returns:
        带元数据的文档
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建文档对象
    doc = Document(
        text=content,
        metadata=metadata
    )

    return doc

# 使用示例
doc = load_with_metadata(
    "data/test_docs/doc1.txt",
    metadata={
        "title": "测试文档1",
        "category": "技术文档",
        "author": "教程作者",
        "date": "2024-01-01"
    }
)

print(f"文档元数据: {doc.metadata}")
```

#### 文件加载器类

```python
# 创建可复用的文档加载器类
class DocumentLoader:
    """文档加载器类"""

    def __init__(self, base_path: str = "data"):
        """
        初始化加载器

        Args:
            base_path: 基础路径
        """
        self.base_path = Path(base_path)

    def load_txt(self, filename: str) -> Document:
        """加载文本文件"""
        filepath = self.base_path / "raw" / filename

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return Document(
            text=content,
            metadata={"source": filename}
        )

    def load_multiple_txt(self, filenames: list) -> list:
        """批量加载文本文件"""
        documents = []
        for filename in filenames:
            try:
                doc = self.load_txt(filename)
                documents.append(doc)
            except Exception as e:
                print(f"加载失败 {filename}: {e}")

        return documents

# 使用示例
loader = DocumentLoader(base_path="data")
docs = loader.load_multiple_txt(["doc1.txt", "doc2.txt"])
print(f"加载了 {len(docs)} 个文档")
```

### 支持的文件格式

LlamaIndex支持150+种数据源，常用格式包括：

| 格式 | 加载器 | 说明 |
|------|--------|------|
| **文本** | SimpleDirectoryReader | .txt, .md |
| **PDF** | PyPDFReader, PDFMinerReader | .pdf |
| **Word** | DocxReader | .docx |
| **网页** | SimpleWebPageReader | HTML |
| **Markdown** | MarkdownReader | .md |
| **CSV** | CSVReader | .csv |

---

## 3.2 文本分块策略

### 为什么需要分块？

**问题**：文档太长，无法一次处理

```
长文档（10,000字）
    ↓
直接输入LLM
    ↓
问题：
1. 超出上下文限制
2. 检索不精确
3. 丢失重要信息
```

**解决方案**：分块处理

```
长文档
    ↓
分块（每块500字）
    ↓
检索最相关的块
    ↓
精准、高效
```

### 分块策略对比

#### 1. 固定长度分块

**原理**：按固定字符数切分

```python
# 固定长度分块
from llama_index.core.node_parser import SentenceSplitter

def split_by_fixed_size(documents, chunk_size=500, chunk_overlap=50):
    """
    固定长度分块

    Args:
        documents: 文档列表
        chunk_size: 每块大小
        chunk_overlap: 重叠大小

    Returns:
        分块列表
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "  # 按空格分词
    )

    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
```

**特点**：
- ✅ 简单快速
- ✅ 可控块大小
- ❌ 可能切断语义

#### 2. 句子分块

**原理**：按句子边界切分

```python
def split_by_sentence(documents, chunk_size=500, chunk_overlap=50):
    """
    按句子分块

    Args:
        documents: 文档列表
        chunk_size: 每块大小
        chunk_overlap: 重叠大小

    Returns:
        分块列表
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="。"  # 中文句号
    )

    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
```

**特点**：
- ✅ 保持句子完整
- ✅ 语义相对完整
- ❌ 可能产生过长/过短的块

#### 3. 段落分块（推荐）

**原理**：优先按段落切分

```python
def split_by_paragraph(documents, chunk_size=1000, chunk_overlap=100):
    """
    按段落分块

    Args:
        documents: 文档列表
        chunk_size: 每块大小
        chunk_overlap: 重叠大小

    Returns:
        分块列表
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"  # 优先按段落分割
    )

    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
```

**特点**：
- ✅ 保持段落完整
- ✅ 语义连贯性好
- ✅ 适合结构化文档
- ⭐ **最推荐用于基础RAG**

### 分块参数选择

#### chunk_size（块大小）的影响

```python
# 实验不同chunk_size的影响
def test_chunk_sizes(documents):
    """
    测试不同chunk_size的效果

    Args:
        documents: 文档列表
    """
    sizes = [200, 500, 1000, 2000]

    for size in sizes:
        nodes = split_by_paragraph(documents, chunk_size=size)
        print(f"chunk_size={size}: {len(nodes)} 个块")

        # 查看第一个块
        if nodes:
            print(f"  第1块长度: {len(nodes[0].text)}")
            print(f"  第1块预览: {nodes[0].text[:100]}...")
        print()
```

**选择建议**：

| 场景 | 推荐chunk_size | 原因 |
|------|---------------|------|
| **短问答** | 300-500 | 信息密集，精准 |
| **长文档分析** | 1000-1500 | 保持上下文 |
| **代码文档** | 500-800 | 保持代码完整 |
| **通用场景** | 500-1000 | 平衡选择 |

#### chunk_overlap（重叠大小）的影响

```
无重叠：
块1: [A B C D]
块2: [E F G H]
→ 可能丢失边界信息

有重叠：
块1: [A B C D]
块2: [D E F G]  ← 'D'重叠
块3: [G H I J]  ← 'G'重叠
→ 保持上下文连续
```

**选择建议**：
- 通常设置为chunk_size的10-20%
- 例如：chunk_size=1000, overlap=100-200

### 完整分块示例

```python
# 文件名：03_02_text_splitting.py
"""
文本分块完整示例
"""

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter

def load_and_split():
    """加载文档并分块"""

    # 1. 加载文档
    print("步骤1: 加载文档")
    reader = SimpleDirectoryReader(
        input_dir="data/processed",
        required_exts=[".txt", ".md"]
    )
    documents = reader.load_data()
    print(f"加载了 {len(documents)} 个文档\n")

    # 2. 测试不同分块策略
    print("步骤2: 测试分块策略\n")

    strategies = [
        ("固定长度(500字)", 500, 50),
        ("段落优先(1000字)", 1000, 100),
        ("大块(2000字)", 2000, 200)
    ]

    results = []

    for name, chunk_size, overlap in strategies:
        print(f"策略: {name}")
        print(f"  chunk_size={chunk_size}, overlap={overlap}")

        # 分块
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator="\n\n"
        )
        nodes = splitter.get_nodes_from_documents(documents)

        # 统计
        print(f"  生成块数: {len(nodes)}")

        # 计算平均长度
        avg_length = sum(len(node.text) for node in nodes) / len(nodes)
        print(f"  平均长度: {avg_length:.0f} 字符")

        # 显示示例
        if nodes:
            print(f"  第1块预览:")
            preview = nodes[0].text[:150]
            print(f"    {preview}...")

        print()
        results.append((name, nodes))

    # 3. 返回最佳策略的结果
    # 通常段落优先是最佳选择
    best_nodes = results[1][1]  # 段落优先策略

    print("="*60)
    print(f"选择策略: {results[1][0]}")
    print(f"生成块数: {len(best_nodes)}")
    print("="*60)

    return best_nodes

def visualize_chunks(nodes):
    """可视化分块结果"""

    print("\n分块可视化:")
    print("="*60)

    for i, node in enumerate(nodes[:5], 1):  # 只显示前5个
        print(f"\n块 {i}:")
        print(f"  长度: {len(node.text)} 字符")
        print(f"  内容预览: {node.text[:100]}...")
        print(f"  元数据: {node.metadata}")

if __name__ == "__main__":
    # 分块
    nodes = load_and_split()

    # 可视化
    visualize_chunks(nodes)

    print("\n✓ 分块完成！")
```

---

## 3.3 文本向量化

### 嵌入模型原理

#### 什么是文本嵌入？

将文本转换为数值向量，使计算机能"理解"语义。

```
文本: "人工智能很强大"
    ↓
嵌入模型
    ↓
向量: [0.23, -0.45, 0.67, ..., 0.12]  (768维)
```

#### 为什么使用嵌入？

**核心思想**：语义相似的文本，在向量空间中距离更近

```
向量空间（简化为2D）：

      猫 🐱
      /|\
     / | \
 狗 🐶  |  老虎🐯
        |
   电脑 💻

"猫" 和 "狗" 距离近 → 语义相似
"电脑" 和 "猫" 距离远 → 语义不同
```

### OpenAI嵌入模型

#### 基础用法

```python
# 文件名：03_03_embeddings.py
"""
文本嵌入示例
"""

from openai import OpenAI
import numpy as np

# 初始化OpenAI客户端
client = OpenAI()  # 自动读取OPENAI_API_KEY环境变量

def get_embedding(text: str, model="text-embedding-3-small"):
    """
    获取文本的嵌入向量

    Args:
        text: 输入文本
        model: 嵌入模型名称

    Returns:
        嵌入向量（numpy数组）
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )

    embedding = response.data[0].embedding
    return np.array(embedding)

def batch_get_embeddings(texts: list, model="text-embedding-3-small"):
    """
    批量获取嵌入向量（更高效）

    Args:
        texts: 文本列表
        model: 模型名称

    Returns:
        嵌入向量列表
    """
    response = client.embeddings.create(
        model=model,
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

def compute_similarity(query: str, documents: list):
    """
    计算查询和文档的相似度

    Args:
        query: 查询文本
        documents: 文档列表

    Returns:
        排序后的(文档, 相似度)列表
    """
    # 生成嵌入
    query_embedding = get_embedding(query)
    doc_embeddings = batch_get_embeddings(documents)

    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    # 排序
    results = list(zip(documents, similarities))
    results.sort(key=lambda x: x[1], reverse=True)

    return results

# 使用示例
if __name__ == "__main__":
    # 示例文档
    documents = [
        "Python是一种高级编程语言",
        "Java也是一种编程语言",
        "今天天气很好",
        "我喜欢吃苹果"
    ]

    print("="*60)
    print("文本嵌入示例")
    print("="*60 + "\n")

    # 1. 生成嵌入
    print("步骤1: 生成嵌入向量")
    embeddings = batch_get_embeddings(documents)
    print(f"嵌入维度: {embeddings.shape[1]}")
    print(f"向量数量: {embeddings.shape[0]}")

    # 显示第一个向量的一部分
    print(f"\n第一个向量的前10个值:")
    print(embeddings[0][:10])
    print()

    # 2. 计算相似度
    print("步骤2: 计算文档相似度\n")

    query = "编程语言有哪些？"
    results = compute_similarity(query, documents)

    print(f"查询: {query}\n")
    print("最相关的文档:")
    for i, (doc, score) in enumerate(results[:3], 1):
        print(f"{i}. 相似度 {score:.3f}: {doc}")

    # 3. 可视化（简化版）
    print("\n步骤3: 相似度可视化")
    print("="*60)

    for doc, score in results:
        bar = "█" * int(score * 50)
        print(f"{score:.3f} {bar} {doc[:30]}...")

    print("\n✓ 嵌入示例完成！")
```

### 开源嵌入模型

#### BGE模型（中文优化）

```python
# 使用开源BGE模型
from sentence_transformers import SentenceTransformer

def load_bge_model():
    """加载BGE模型（首次会自动下载）"""
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    return model

def embed_with_bge(texts: list):
    """
    使用BGE模型生成嵌入

    Args:
        texts: 文本列表

    Returns:
        嵌入向量
    """
    model = load_bge_model()
    embeddings = model.encode(texts)
    return embeddings

# 使用示例
if __name__ == "__main__":
    texts = ["Python是编程语言", "Java也是编程语言"]
    embeddings = embed_with_bge(texts)
    print(f"嵌入形状: {embeddings.shape}")
```

**模型选择对比**：

| 模型 | 维度 | 速度 | 质量 | 成本 | 推荐场景 |
|------|------|------|------|------|---------|
| **OpenAI small** | 1536 | 快 | 优秀 | 付费 | 通用、快速上线 |
| **OpenAI large** | 3072 | 中 | 最优 | 付费 | 高精度需求 |
| **BGE-small** | 512 | 很快 | 良好 | 免费 | 中文、私有部署 |
| **BGE-large** | 1024 | 中 | 优秀 | 免费 | 中文高质量 |

---

## 3.4 向量检索

### 向量数据库基础

#### 为什么需要向量数据库？

**问题**：如何高效搜索百万级向量？

```
朴素方法：
  对于每个查询，遍历所有向量计算相似度
  → 时间复杂度O(n)，太慢！

向量数据库：
  使用索引算法（HNSW、IVF等）
  → 时间复杂度O(log n)，快很多！
```

### Chroma快速入门

#### 基础操作

```python
# 文件名：03_04_vector_retrieval.py
"""
向量检索示例
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np

class VectorStore:
    """向量存储类"""

    def __init__(self, collection_name="rag_documents"):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
        """
        # 创建持久化客户端
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

        # OpenAI客户端（用于生成嵌入）
        self.openai_client = OpenAI()

    def add_documents(self, texts: list, metadatas: list = None):
        """
        添加文档到向量库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        # 生成嵌入
        embeddings = self._get_embeddings(texts)

        # 生成ID
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 添加到集合
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✓ 添加了 {len(texts)} 个文档")

    def query(self, query_text: str, n_results: int = 3):
        """
        查询相似文档

        Args:
            query_text: 查询文本
            n_results: 返回结果数量

        Returns:
            查询结果
        """
        # 生成查询嵌入
        query_embedding = self._get_embeddings([query_text])[0]

        # 查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def _get_embeddings(self, texts: list):
        """获取嵌入向量"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings

# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("向量检索示例")
    print("="*60 + "\n")

    # 1. 创建向量存储
    print("步骤1: 创建向量存储")
    vector_store = VectorStore("demo_collection")
    print()

    # 2. 添加文档
    print("步骤2: 添加文档")
    documents = [
        "Python是一种高级编程语言，由Guido创建",
        "JavaScript主要用于Web前端开发",
        "Rust注重内存安全和性能",
        "Go语言适合并发编程",
        "Java是企业级开发的常用语言"
    ]

    metadatas = [
        {"category": "编程语言", "name": "Python"},
        {"category": "编程语言", "name": "JavaScript"},
        {"category": "编程语言", "name": "Rust"},
        {"category": "编程语言", "name": "Go"},
        {"category": "编程语言", "name": "Java"}
    ]

    vector_store.add_documents(documents, metadatas)
    print()

    # 3. 查询
    print("步骤3: 查询相似文档\n")

    queries = [
        "什么语言性能最好？",
        "如何做Web开发？",
        "适合系统的语言"
    ]

    for query in queries:
        print(f"查询: {query}")
        results = vector_store.query(query, n_results=2)

        print("最相关的文档:")
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ), 1):
            similarity = 1 - distance  # 转换为相似度
            print(f"  {i}. {doc}")
            print(f"     相似度: {similarity:.3f}")
            print(f"     元数据: {metadata}")
        print()

    print("="*60)
    print("✓ 向量检索示例完成！")
    print("="*60)
```

### 检索质量评估

```python
def evaluate_retrieval(vector_store, test_queries, expected_docs):
    """
    评估检索质量

    Args:
        vector_store: 向量存储对象
        test_queries: 测试查询列表
        expected_docs: 期望的文档索引

    Returns:
        准确率
    """
    correct = 0
    total = len(test_queries)

    for query, expected_idx in zip(test_queries, expected_docs):
        results = vector_store.query(query, n_results=3)
        retrieved_idx = int(results['ids'][0][0].split('_')[1])

        if retrieved_idx == expected_idx:
            correct += 1

        print(f"查询: {query}")
        print(f"  期望: {expected_idx}, 实际: {retrieved_idx}, {'✓' if retrieved_idx == expected_idx else '✗'}")

    accuracy = correct / total
    print(f"\n准确率: {accuracy:.2%}")
    return accuracy
```

---

## 3.5 LLM生成回答

### RAG生成流程

```python
# 文件名：03_05_rag_generation.py
"""
完整RAG生成示例
"""

from openai import OpenAI
from vector_store import VectorStore

class RAGSystem:
    """简单的RAG系统"""

    def __init__(self, collection_name="rag_documents"):
        """
        初始化RAG系统

        Args:
            collection_name: 向量集合名称
        """
        self.vector_store = VectorStore(collection_name)
        self.llm_client = OpenAI()

    def add_documents(self, documents: list):
        """
        添加文档到知识库

        Args:
            documents: 文档列表
        """
        self.vector_store.add_documents(documents)

    def query(self, question: str, n_results: int = 3):
        """
        查询RAG系统

        Args:
            question: 用户问题
            n_results: 检索文档数量

        Returns:
            答案和来源
        """
        # 1. 检索相关文档
        print(f"\n步骤1: 检索相关文档")
        retrieval_results = self.vector_store.query(question, n_results)

        if not retrieval_results['documents'][0]:
            return "抱歉，知识库中没有找到相关信息。", []

        context_docs = retrieval_results['documents'][0]
        print(f"检索到 {len(context_docs)} 个相关文档")

        # 2. 构建提示词
        print(f"步骤2: 构建提示词")
        context = "\n\n".join([
            f"【文档{i+1}】\n{doc}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""
你是一个专业的问答助手。请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

要求：
1. 基于文档内容回答
2. 如果文档中没有相关信息，明确说明
3. 回答要准确、简洁
4. 引用参考的文档编号

回答：
"""

        # 3. 生成答案
        print(f"步骤3: 生成答案")
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return answer, context_docs

# 完整示例
if __name__ == "__main__":
    print("="*70)
    print("完整RAG系统示例")
    print("="*70)

    # 1. 创建RAG系统
    print("\n初始化RAG系统...")
    rag = RAGSystem("demo_rag")

    # 2. 添加知识库
    print("\n添加知识库...")
    knowledge_base = [
        {
            "text": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。"
                   "Python的特点是语法简洁、易学易用、应用广泛。",
            "metadata": {"topic": "Python介绍"}
        },
        {
            "text": "Python可用于Web开发（Django、Flask）、数据分析（Pandas、NumPy）、"
                   "人工智能（TensorFlow、PyTorch）等多个领域。",
            "metadata": {"topic": "Python应用"}
        },
        {
            "text": "JavaScript是一种脚本语言，主要用于Web前端开发。"
                   "它可以创建动态的网页内容，与HTML和CSS一起构成Web的三大核心技术。",
            "metadata": {"topic": "JavaScript介绍"}
        },
        {
            "text": "Rust是一种系统编程语言，注重内存安全、并发和性能。"
                   "它没有垃圾回收，而是通过所有权系统在编译时保证内存安全。",
            "metadata": {"topic": "Rust介绍"}
        }
    ]

    documents = [item["text"] for item in knowledge_base]
    rag.add_documents(documents)

    # 3. 测试查询
    test_questions = [
        "Python有什么特点？",
        "JavaScript主要用于什么？",
        "Rust如何保证内存安全？",
        "Python可以用来做什么？"  # 需要综合多个文档
    ]

    for question in test_questions:
        print("\n" + "="*70)
        print(f"用户问题: {question}")
        print("="*70)

        answer, sources = rag.query(question)

        print("\n【答案】")
        print(answer)

        print("\n【参考来源】")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source[:100]}...")

    print("\n" + "="*70)
    print("✓ RAG系统演示完成！")
    print("="*70)
```

### 完整RAG项目模板

```python
# 文件名：simple_rag_system.py
"""
可复用的简单RAG系统
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM
import chromadb

# 加载环境变量
load_dotenv()

class SimpleRAG:
    """简单RAG系统"""

    def __init__(self, data_path="data/processed", persist_dir="./chroma_db"):
        """
        初始化RAG系统

        Args:
            data_path: 文档路径
            persist_dir: 向量库持久化目录
        """
        self.data_path = data_path
        self.persist_dir = persist_dir
        self.index = None

    def load_documents(self):
        """加载文档"""
        reader = SimpleDirectoryReader(self.data_path)
        documents = reader.load_data()
        print(f"✓ 加载了 {len(documents)} 个文档")
        return documents

    def build_index(self, documents):
        """
        构建向量索引

        Args:
            documents: 文档列表
        """
        # 分块
        splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"✓ 分块生成 {len(nodes)} 个节点")

        # 创建向量库
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        collection = chroma_client.get_or_create_collection("rag_documents")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 构建索引
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        )
        print("✓ 向量索引构建完成")

    def query(self, question: str, similarity_threshold: float = 0.7):
        """
        查询RAG系统

        Args:
            question: 用户问题
            similarity_threshold: 相似度阈值

        Returns:
            答案
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index()")

        # 创建查询引擎
        query_engine = self.index.as_query_engine(
            llm=OpenAILLM(model="gpt-3.5-turbo", temperature=0.7),
            similarity_top_k=3,
            vector_store_query_mode="default"
        )

        # 查询
        response = query_engine.query(question)

        return response

# 使用示例
if __name__ == "__main__":
    # 创建RAG系统
    rag = SimpleRAG()

    # 加载文档
    documents = rag.load_documents()

    # 构建索引
    rag.build_index(documents)

    # 查询
    while True:
        question = input("\n输入问题（或'quit'退出）: ").strip()

        if question.lower() == 'quit':
            break

        if not question:
            continue

        response = rag.query(question)
        print(f"\n答案:\n{response}")
```

---

## 总结

### 本章要点回顾

1. **文档加载**：使用SimpleDirectoryReader加载多种格式文档
2. **文本分块**：按段落分块，保持语义完整
3. **文本向量化**：使用OpenAI embeddings或开源模型
4. **向量检索**：使用Chroma进行高效相似度搜索
5. **LLM生成**：基于检索到的文档生成答案

### 学习检查清单

- [ ] 能够加载本地文档
- [ ] 理解不同分块策略的效果
- [ ] 掌握嵌入模型的使用
- [ ] 能够构建向量检索系统
- [ ] 实现了完整的RAG问答流程
- [ ] 理解RAG的完整工作流程

### 下一步学习

- **下一章**：[第4章：RAG评估基础](./04-RAG评估基础.md)
- **实战项目**：完成第5章的综合项目

### 扩展练习

1. **基础练习**：
   - 加载自己的文档并测试
   - 尝试不同的分块参数
   - 比较OpenAI和BGE嵌入效果

2. **进阶练习**：
   - 实现混合检索（向量+关键词）
   - 添加文档来源追踪
   - 优化提示词模板

3. **挑战项目**：
   - 构建一个多轮对话RAG系统
   - 实现查询重写功能

---

## 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| **分块** | Chunking | 将长文档切分成小块 |
| **嵌入** | Embedding | 文本到向量的转换 |
| **向量空间** | Vector Space | 嵌入向量的多维空间 |
| **余弦相似度** | Cosine Similarity | 衡量向量相似度的指标 |
| **索引** | Index | 加速检索的数据结构 |

---

**返回目录** | **上一章：环境搭建** | **下一章：RAG评估基础**

---

**本章结束**

> 恭喜你完成了第一个RAG系统！这是整个教程的重要里程碑。接下来我们将学习如何评估和优化RAG系统，让它的性能更上一层楼。
