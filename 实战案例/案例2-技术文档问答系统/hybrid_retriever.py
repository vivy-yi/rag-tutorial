"""
案例2：混合检索器实现
"""

from typing import List, Dict, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import jieba
except ImportError:
    print("警告：部分依赖未安装，将使用简化实现")


class HybridRetriever:
    """混合检索器：向量检索 + BM25"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.documents = []
        self.embeddings = []
        self.bm25 = None
        self.doc_count = 0
        self.code_count = 0

        try:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except:
            self.model = None
            print("使用简化检索模式")

    def add_documents(self, documents: List[Dict]):
        """添加文档"""
        self.documents = documents

        # 统计
        self.doc_count = len([d for d in documents if d['metadata']['type'] == 'text'])
        self.code_count = len([d for d in documents if d['metadata']['type'] == 'code'])

        # 生成嵌入
        if self.model:
            texts = [d['content'] for d in documents]
            self.embeddings = self.model.encode(texts)

        # 构建BM25索引
        try:
            tokenized_docs = [list(jieba.cut(d['content'])) for d in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        except:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> List[Dict]:
        """检索文档"""

        results = []

        if mode == "vector" and self.model:
            results = self._vector_search(query, top_k)
        elif mode == "keyword" and self.bm25:
            results = self._bm25_search(query, top_k)
        elif mode == "hybrid":
            results = self._hybrid_search(query, top_k)
        else:
            # 简化实现：关键词匹配
            results = self._simple_search(query, top_k)

        return results

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """向量检索"""
        query_emb = self.model.encode([query])
        scores = np.dot(self.embeddings, query_emb.T).flatten()

        indices = scores.argsort()[-top_k:][::-1]

        return [
            {
                **self.documents[i],
                "score": float(scores[i])
            }
            for i in indices
        ]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        indices = scores.argsort()[-top_k:][::-1]

        return [
            {
                **self.documents[i],
                "score": float(scores[i] / max(scores))
            }
            for i in indices if scores[i] > 0
        ]

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """混合检索"""
        # 向量检索
        if self.model:
            vector_results = self._vector_search(query, top_k * 2)
        else:
            vector_results = []

        # BM25检索
        if self.bm25:
            bm25_results = self._bm25_search(query, top_k * 2)
        else:
            bm25_results = []

        # 合并结果（加权）
        combined = {}

        for doc in vector_results:
            doc_id = doc.get('id', doc['content'][:50])
            combined[doc_id] = {**doc, "vector_score": doc["score"], "bm25_score": 0}

        for doc in bm25_results:
            doc_id = doc.get('id', doc['content'][:50])
            if doc_id in combined:
                combined[doc_id]["bm25_score"] = doc["score"]
            else:
                combined[doc_id] = {**doc, "vector_score": 0, "bm25_score": doc["score"]}

        # 计算综合分数
        for doc_id, doc in combined.items():
            doc["score"] = 0.6 * doc["vector_score"] + 0.4 * doc["bm25_score"]

        # 排序并返回top-k
        sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_docs[:top_k]

    def _simple_search(self, query: str, top_k: int) -> List[Dict]:
        """简化检索"""
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)

            if score > 0:
                results.append({
                    **doc,
                    "score": score / 10.0
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
