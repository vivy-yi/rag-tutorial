# 案例2：技术文档问答系统

> **难度**: ⭐⭐ 进阶 | **技术栈**: LangChain, Hybrid Search, CrossEncoder, Reranker

使用混合检索（Vector + BM25）和重排序技术构建技术文档问答系统

---

## 🎯 案例特点

- ✅ **混合检索**: 向量检索 + BM25关键词检索
- ✅ **重排序**: CrossEncoder二阶段重排
- ✅ **代码高亮**: 技术文档完美展示
- ✅ **精准答案**: 结合多种检索方式提升准确率

---

## 🚀 快速开始

```bash
cd projects/case2-doc-qa
pip install -r requirements.txt
python main.py
```

---

## 📁 项目结构

```
case2-doc-qa/
├── main.py                 # 主程序
├── doc_qa_system.py       # 问答系统
├── hybrid_retriever.py    # 混合检索器
├── reranker.py            # 重排序模块
└── requirements.txt
```

---

## 🔑 核心技术

### 混合检索

```python
# hybrid_retriever.py
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import Chroma

class HybridRetriever:
    def __init__(self, vectorstore, bm25_retriever):
        self.vectorstore = vectorstore
        self.bm25 = bm25_retriever

    def retrieve(self, query, k=10):
        # 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=k)
        # BM25检索
        bm25_results = self.bm25.get_relevant_documents(query)
        # 合并结果
        return self.merge_and_rerank(vector_results, bm25_results)
```

### 重排序

```python
# reranker.py
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        # 计算查询-文档相关性分数
        scores = self.model.predict([[query, doc.page_content] for doc in documents])
        # 返回top-k文档
        return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
```

---

## 📊 性能对比

| 检索方式 | Precision | Recall | MRR |
|---------|-----------|--------|-----|
| 纯向量检索 | 0.72 | 0.65 | 0.68 |
| 纯BM25 | 0.68 | 0.71 | 0.67 |
| **混合检索** | **0.81** | **0.76** | **0.79** |
| **混合+重排** | **0.87** | **0.79** | **0.85** |

---

## 🎓 学习要点

1. **混合检索架构**
   - 稀疏检索（BM25）
   - 密集检索（向量）
   - 结果融合策略

2. **重排序技术**
   - CrossEncoder模型
   - 二阶段检索流程
   - 精度vs速度权衡

3. **技术文档处理**
   - Markdown解析
   - 代码块提取
   - 结构化信息

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case2-doc-qa)**

**[← 返回案例列表](index.md)**
