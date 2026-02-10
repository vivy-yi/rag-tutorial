"""
案例4：GraphRAG系统
"""

from typing import List, Dict, Tuple


class GraphRAGSystem:
    """基于知识图谱的RAG系统"""

    def __init__(self, graph):
        self.graph = graph

    def query(self, question: str) -> Dict:
        """查询知识图谱"""

        # 简化实现：关键词匹配
        question_lower = question.lower()

        # 查找相关实体和关系
        reasoning_path = []
        answer_parts = []

        # 实体查询
        for entity in self.graph["entities"]:
            if entity["name"].lower() in question_lower:
                reasoning_path.append(f"找到实体：{entity['name']}")

                # 查找关系
                for rel in self.graph["relationships"]:
                    if rel["source"] == entity["id"]:
                        target = self._get_entity_name(rel["target"])
                        answer_parts.append(f"{entity['name']}{rel['type']}{target}")

                # 查找属性
                for prop in entity.get("properties", {}).items():
                    answer_parts.append(f"{entity['name']}的{prop[0]}是{prop[1]}")

        # 生成答案
        if answer_parts:
            answer = "、".join(answer_parts) + "。"
        else:
            answer = "抱歉，知识库中没有找到相关信息。"

        return {
            "answer": answer,
            "reasoning_path": reasoning_path
        }

    def _get_entity_name(self, entity_id: str) -> str:
        """获取实体名称"""
        for entity in self.graph["entities"]:
            if entity["id"] == entity_id:
                return entity["name"]
        return entity_id
