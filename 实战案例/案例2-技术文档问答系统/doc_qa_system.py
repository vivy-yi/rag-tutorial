"""
案例2：技术文档问答系统
"""

from typing import List, Dict, Optional

class TechDocQA:
    """技术文档问答系统"""

    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker
        self._load_documents()

    def _load_documents(self):
        """加载技术文档"""
        # 模拟技术文档
        docs = [
            {
                "id": "1",
                "title": "FastAPI入门指南",
                "content": "FastAPI是一个现代、快速的Python Web框架。创建API非常简单：\n\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}",
                "metadata": {"type": "code", "language": "python", "topic": "fastapi"}
            },
            {
                "id": "2",
                "title": "Python列表推导式",
                "content": "列表推导式是Python创建列表的简洁方式。\n\n基本语法：[表达式 for 项 in 可迭代对象]\n\n例如：\nsquares = [x**2 for x in range(10)]\n\n带条件：\nevens = [x for x in range(10) if x % 2 == 0]",
                "metadata": {"type": "code", "language": "python", "topic": "basics"}
            },
            {
                "id": "3",
                "title": "PyTorch神经网络定义",
                "content": "使用PyTorch定义神经网络：\n\nimport torch.nn as nn\n\nclass NeuralNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(784, 128)\n        self.fc2 = nn.Linear(128, 10)\n    \n    def forward(self, x):\n        x = torch.relu(self.fc1(x))\n        x = self.fc2(x)\n        return x",
                "metadata": {"type": "code", "language": "python", "topic": "pytorch"}
            },
            {
                "id": "4",
                "title": "Django MVC架构",
                "content": "Django采用MTV（Model-Template-View）架构：\n\n- Model：数据模型，负责数据库交互\n- Template：HTML模板，负责展示\n- View：业务逻辑，负责处理请求\n\n这种分离使得代码更易维护。",
                "metadata": {"type": "text", "topic": "django"}
            },
            {
                "id": "5",
                "title": "装饰器基础",
                "content": "Python装饰器是一种修改函数行为的强大工具：\n\n@my_decorator\ndef my_function():\n    pass\n\n等价于：my_function = my_decorator(my_function)",
                "metadata": {"type": "text", "topic": "advanced"}
            }
        ]

        self.retriever.add_documents(docs)

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        use_reranking: bool = True
    ) -> Dict:
        """查询"""

        # 检索
        documents = self.retriever.retrieve(question, mode, top_k)

        # 重排序
        if use_reranking:
            documents = self.reranker.rerank(question, documents, top_k)

        # 生成答案
        answer = self._generate_answer(question, documents)

        # 提取相关查询
        related_queries = self._generate_related_queries(question)

        return {
            "answer": answer,
            "documents": documents,
            "confidence": documents[0]['score'] if documents else 0,
            "related_queries": related_queries
        }

    def _generate_answer(self, question: str, documents: List[Dict]) -> str:
        """生成答案"""

        if not documents:
            return "抱歉，没有找到相关文档。"

        top_doc = documents[0]

        if top_doc['metadata']['type'] == 'code':
            return f"找到了相关代码示例：\n\n**{top_doc['title']}**\n\n```python\n{top_doc['content']}\n```\n\n这个示例应该能解决你的问题。"
        else:
            return f"根据技术文档：{top_doc['content'][:300]}..."

    def _generate_related_queries(self, query: str) -> List[str]:
        """生成相关查询"""

        # 简化实现
        related = {
            "FastAPI": ["如何部署FastAPI应用", "FastAPI异步编程"],
            "列表推导": ["字典推导式", "集合推导式"],
            "PyTorch": ["TensorFlow对比", "PyTorch数据加载"],
            "Django": ["Flask对比", "Django ORM使用"],
            "装饰器": ["类装饰器", "装饰器链"]
        }

        for key, queries in related.items():
            if key in query:
                return queries

        return []
