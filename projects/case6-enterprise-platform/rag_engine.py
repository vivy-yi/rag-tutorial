"""
案例6：企业RAG引擎
"""

from typing import List, Dict
import hashlib
import time


class EnterpriseRAGEngine:
    """企业级RAG引擎"""

    def __init__(self):
        self.documents = {}
        self.user_documents = {}
        self.query_stats = {}

    def add_document(self, content: str, filename: str, user_id: str) -> str:
        """添加文档"""
        doc_id = hashlib.md5(f"{user_id}:{filename}:{time.time()}".encode()).hexdigest()

        self.documents[doc_id] = {
            "content": content,
            "filename": filename,
            "user_id": user_id,
            "created_at": time.time()
        }

        if user_id not in self.user_documents:
            self.user_documents[user_id] = []
        self.user_documents[user_id].append(doc_id)

        return doc_id

    def query(self, question: str, user_id: str, top_k: int = 5) -> Dict:
        """查询"""
        # 记录统计
        self._record_query(user_id)

        # 获取用户文档
        user_doc_ids = self.user_documents.get(user_id, [])

        # 简化检索：关键词匹配
        results = []
        question_lower = question.lower()

        for doc_id in user_doc_ids:
            doc = self.documents[doc_id]
            content = doc["content"].lower()

            score = sum(1 for word in question_lower.split() if word in content)

            if score > 0:
                results.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "content": doc["content"][:500],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        # 生成答案
        if top_results:
            answer = f"找到{len(top_results)}个相关文档。{top_results[0]['content'][:200]}..."
            sources = [r["filename"] for r in top_results]
            confidence = min(0.9, top_results[0]["score"] * 0.1)
        else:
            answer = "没有找到相关文档。"
            sources = []
            confidence = 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    def list_user_documents(self, user_id: str) -> List[Dict]:
        """列出用户文档"""
        doc_ids = self.user_documents.get(user_id, [])
        return [
            {
                "id": doc_id,
                "filename": self.documents[doc_id]["filename"],
                "created_at": self.documents[doc_id]["created_at"]
            }
            for doc_id in doc_ids
        ]

    def get_stats(self, user_id: str) -> Dict:
        """获取统计"""
        return {
            "total_documents": len(self.user_documents.get(user_id, [])),
            "total_queries": self.query_stats.get(user_id, 0),
            "cache_hit_rate": 0.75  # 模拟数据
        }

    def _record_query(self, user_id: str):
        """记录查询统计"""
        if user_id not in self.query_stats:
            self.query_stats[user_id] = 0
        self.query_stats[user_id] += 1
