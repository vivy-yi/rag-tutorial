"""
案例5：多模态RAG系统
"""

from typing import List, Dict, Optional
from PIL import Image
import numpy as np


class MultimodalRAG:
    """多模态RAG系统"""

    def __init__(self):
        self.product_database = self._load_products()

    def _load_products(self) -> List[Dict]:
        """加载产品数据库"""
        return [
            {
                "id": "p1",
                "name": "iPhone 15 Pro",
                "description": "苹果最新旗舰，钛金属边框，A17芯片",
                "category": "手机",
                "price": 7999,
                "features": ["5G", "面容ID", "三摄像头"]
            },
            {
                "id": "p2",
                "name": "MacBook Pro 14",
                "description": "M3 Pro芯片，专业级性能",
                "category": "电脑",
                "price": 15999,
                "features": ["M3芯片", "Liquid视网膜XDR屏", "18小时续航"]
            },
            {
                "id": "p3",
                "name": "AirPods Pro 2",
                "description": "主动降噪，空间音频",
                "category": "耳机",
                "price": 1899,
                "features": ["主动降噪", "通透模式", "MagSafe充电"]
            }
        ]

    def query(
        self,
        image: Optional[Image.Image] = None,
        text: str = "",
        mode: str = "hybrid"
    ) -> Dict:
        """多模态查询"""

        results = []
        confidence = 0.0

        # 图像分析（简化）
        if image and mode in ["图文结合", "仅图像"]:
            image_features = self._analyze_image(image)
            results.extend(image_features)
            confidence = max(confidence, 0.7)

        # 文本搜索
        if text and mode in ["图文结合", "仅文本"]:
            text_results = self._search_products(text)
            results.extend(text_results)
            confidence = max(confidence, 0.8)

        # 去重并排序
        unique_results = self._deduplicate(results)

        # 生成答案
        answer = self._generate_answer(text, unique_results)

        return {
            "answer": answer,
            "products": unique_results[:3],
            "confidence": confidence
        }

    def _analyze_image(self, image: Image.Image) -> List[Dict]:
        """分析图像（简化实现）"""
        # 实际应用中应该使用CLIP或GPT-4V
        # 这里简化为返回所有产品
        return self.product_database

    def _search_products(self, query: str) -> List[Dict]:
        """文本搜索"""
        query_lower = query.lower()
        results = []

        for product in self.product_database:
            score = 0
            if query_lower in product["name"].lower():
                score += 3
            if query_lower in product["description"].lower():
                score += 2
            for feature in product["features"]:
                if query_lower in feature.lower():
                    score += 1

            if score > 0:
                results.append({**product, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        unique = []
        for item in results:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)
        return unique

    def _generate_answer(self, query: str, products: List[Dict]) -> str:
        """生成答案"""
        if not products:
            return "抱歉，没有找到相关产品。"

        top_product = products[0]
        return f"为您找到：{top_product['name']}，{top_product['description']}，价格¥{top_product['price']}。"
