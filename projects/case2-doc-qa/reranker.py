"""
案例2：重排序器
"""

from typing import List, Dict

class CrossEncoderReranker:
    """交叉编码器重排序器"""

    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('ms-marco-MiniLM-L-6-v2')
        except:
            self.model = None
            print("使用简化重排序")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """重排序"""

        if not documents:
            return documents

        if self.model:
            return self._cross_encoder_rerank(query, documents, top_k)
        else:
            return self._simple_rerank(query, documents, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """使用CrossEncoder重排序"""

        # 准备输入
        pairs = [[query, doc['content'][:512]] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 更新分数
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # 排序
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)

        # 返回top-k
        if top_k:
            return documents[:top_k]
        return documents

    def _simple_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """简化重排序：基于关键词密度"""

        query_words = set(query.lower().split())

        for doc in documents:
            content_words = set(doc['content'].lower().split())
            overlap = len(query_words & content_words)
            doc['rerank_score'] = overlap / max(len(query_words), 1)

        documents.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k:
            return documents[:top_k]
        return documents
