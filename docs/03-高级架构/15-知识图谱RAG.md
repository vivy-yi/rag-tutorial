# 第15章：知识图谱RAG

> 超越扁平检索：利用知识图谱的结构化信息和关系推理能力，实现更深层次的知识问答！

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解知识图谱的基本概念
- [ ] 掌握GraphRAG的核心原理
- [ ] 实现图谱增强的RAG系统
- [ ] 应用图检索算法
- [ ] 构建知识图谱问答系统

**预计学习时间**：6小时
**难度等级**：⭐⭐⭐⭐⭐

---

## 前置知识

- [ ] 完成模块1和模块2
- [ ] 了解图数据库基础
- [ ] 理解实体关系抽取
- [ ] 熟悉NetworkX或Neo4j

---

## 15.1 为什么需要知识图谱？

### 15.1.1 向量检索的局限

```
场景：多跳推理问题

问题："马斯克的公司最近的收购是什么？"

向量检索的问题：
  Step 1: 检索"马斯克 收购"
    → 可能找到相关文档
    → 但无法理解"马斯克→公司→收购"的关系链

  Step 2: 需要多次检索
    → "马斯克的公司"
    → "SpaceX收购历史"
    → 等等...

  Step 3: 上下文丢失
    → 每次检索都是独立的
    → 无法维护关系链

知识图谱的优势：
  ✓ 结构化知识（实体-关系-实体）
  ✓ 关系推理（直接遍历图）
  ✓ 多跳能力（1跳→2跳→3跳）
  ✓ 可解释性（清晰的推理路径）
```

### 15.1.2 GraphRAG原理

```
GraphRAG = 知识图谱 + RAG

传统RAG：
  Query → 向量检索 → 文档 → Answer

GraphRAG：
  Query → 实体识别 → 图谱检索 → 子图 → LLM → Answer

核心组件：
  1. 实体识别：从查询中提取实体
  2. 图谱检索：找到相关子图
  3. 图嵌入：将子图转换为向量
  4. 答案生成：基于子图生成答案
```

---

## 15.2 知识图谱基础

### 15.2.1 图数据结构

```python
# 文件名：knowledge_graph.py
"""
知识图谱实现
"""

from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt


class KnowledgeGraph:
    """
    知识图谱

    基于NetworkX实现

    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.add_entity("SpaceX", {"type": "Company", "founded": 2002})
        >>> kg.add_entity("Elon Musk", {"type": "Person"})
        >>> kg.add_relation("Elon Musk", "CEO", "SpaceX")
        >>> kg.visualize()
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, attributes: Dict):
        """
        添加实体

        Args:
            entity_id: 实体ID（唯一标识符）
            attributes: 实体属性字典
        """
        self.graph.add_node(entity_id, **attributes)

    def add_relation(self, entity1: str, relation: str, entity2: str,
                    attributes: Dict = None):
        """
        添加关系

        Args:
            entity1: 源实体
            relation: 关系类型
            entity2: 目标实体
            attributes: 关系属性（可选）
        """
        if attributes is None:
            attributes = {}

        self.graph.add_edge(entity1, entity2, relation=relation, **attributes)

    def get_neighbors(self, entity: str, direction: str = "out") -> List[Tuple[str, str]]:
        """
        获取邻居实体

        Args:
            entity: 实体ID
            direction: "out"（出边）或"in"（入边）或"both"

        Returns:
            [(neighbor_id, relation), ...]
        """
        if direction == "out":
            neighbors = self.graph.out_edges(entity, data=True)
        elif direction == "in":
            neighbors = self.graph.in_edges(entity, data=True)
        else:
            neighbors = self.graph.edges(entity, data=True)

        return [(v if direction == "in" else u,
                edge['relation'])
                for u, v, edge in neighbors]

    def find_path(self, entity1: str, entity2: str) -> List[str]:
        """
        查找两实体间最短路径

        Returns:
            [entity1, ..., entity2]
        """
        try:
            path = nx.shortest_path(self.graph, entity1, entity2)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_subgraph(self, entities: List[str], hops: int = 2) -> nx.DiGraph:
        """
        获取子图（包含指定实体及其邻域）

        Args:
            entities: 中心实体列表
            hops: 跳数

        Returns:
            子图
        """
        subgraph_nodes = set(entities)

        # 扩展邻域
        for _ in range(hops):
            new_nodes = set()
            for node in subgraph_nodes:
                neighbors = self.graph.neighbors(node)
                new_nodes.update(neighbors)
            subgraph_nodes.update(new_nodes)

        # 提取子图
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        return subgraph

    def visualize(self, figsize=(12, 8)):
        """可视化图谱"""
        plt.figure(figsize=figsize)

        # 使用spring布局
        pos = nx.spring_layout(self.graph, k=1, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos,
                               node_color='lightblue',
                               node_size=500)

        # 绘制边
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowstyle='->',
                              arrowsize=20)

        # 绘制标签
        labels = nx.get_node_attributes(self.graph, 'type')
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)

        plt.title("Knowledge Graph")
        plt.axis('off')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建知识图谱
    kg = KnowledgeGraph()

    # 添加实体
    kg.add_entity("SpaceX", {"type": "Company", "founded": 2002})
    kg.add_entity("Tesla", {"type": "Company", "founded": 2003})
    kg.add_entity("Elon Musk", {"type": "Person", "born": 1971})
    kg.add_entity("Starship", {"type": "Rocket", "status": "development"})

    # 添加关系
    kg.add_relation("Elon Musk", "CEO", "SpaceX")
    kg.add_relation("Elon Musk", "CEO", "Tesla")
    kg.add_relation("SpaceX", "develops", "Starship")

    # 查询
    print("SpaceX的邻居:")
    for neighbor, relation in kg.get_neighbors("SpaceX"):
        print(f"  {relation} → {neighbor}")

    # 查找路径
    path = kg.find_path("Tesla", "Starship")
    print(f"\nTesla到Starship的路径: {path}")

    # 可视化
    kg.visualize()
```

---

## 15.3 GraphRAG实现

### 15.3.1 核心流程

```python
# 文件名：graph_rag.py
"""
GraphRAG系统实现
"""

from typing import List, Dict, Any
import numpy as np


class GraphRAGSystem:
    """
    GraphRAG系统

    结合知识图谱和向量检索
    """

    def __init__(self, knowledge_graph: 'KnowledgeGraph',
                 vector_store, llm_client):
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.llm = llm_client

    def query(self, query: str, use_graph: bool = True) -> Dict:
        """
        GraphRAG查询

        Args:
            query: 用户查询
            use_graph: 是否使用图谱
        """
        if not use_graph:
            # 传统RAG
            return self._vector_rag(query)

        # 步骤1：实体识别
        entities = self._extract_entities(query)
        print(f"识别到的实体: {entities}")

        # 步骤2：图谱检索
        subgraph = self._graph_retrieve(entities)
        print(f"子图包含 {subgraph.number_of_nodes()} 个节点")

        # 步骤3：增强检索
        graph_context = self._get_graph_context(subgraph)
        vector_context = self._vector_retrieve(query)

        # 步骤4：融合上下文
        combined_context = self._combine_contexts(graph_context, vector_context)

        # 步骤5：生成答案
        answer = self._generate_answer(query, combined_context, subgraph)

        return {
            'answer': answer,
            'entities': entities,
            'subgraph_nodes': subgraph.number_of_nodes(),
            'graph_used': True
        }

    def _extract_entities(self, query: str) -> List[str]:
        """
        实体识别（简化版）

        实际使用时应使用NER模型或知识图谱链接
        """
        # 简化示例：基于规则提取
        entities = []

        # 大写字母开头的词（可能是专有名词）
        import re
        potential_entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        entities.extend(potential_entities)

        # 去重
        entities = list(set(entities))

        return entities

    def _graph_retrieve(self, entities: List[str]) -> 'KnowledgeGraph':
        """
        图谱检索：获取相关子图
        """
        all_nodes = set(entities)

        # 扩展1跳
        for entity in entities:
            if entity in self.kg.graph:
                neighbors = self.kg.graph.neighbors(entity)
                all_nodes.update(neighbors)

        # 提取子图
        subgraph = self.kg.graph.subgraph(all_nodes).copy()

        return KnowledgeGraph.from_networkx(subgraph)

    def _get_graph_context(self, subgraph: 'KnowledgeGraph') -> str:
        """
        将子图转换为文本上下文
        """
        contexts = []

        for edge in subgraph.graph.edges(data=True):
            u, v, data = edge
            contexts.append(f"{u} -> {data.get('relation', 'related to')} -> {v}")

        for node in subgraph.graph.nodes(data=True):
            node_id, node_data = node
            attrs = ", ".join([f"{k}={v}" for k, v in node_data.items()])
            contexts.append(f"{node_id}: {attrs}")

        return "\n".join(contexts)

    def _vector_rag(self, query: str) -> Dict:
        """传统向量RAG"""
        # 向量检索
        results = self.vector_store.search(query, top_k=5)

        # 生成答案
        context = "\n".join([r['text'] for r in results])
        answer = self.llm.generate(f"基于上下文回答问题：{query}\n\n上下文：{context}")

        return {
            'answer': answer,
            'graph_used': False
        }

    def _vector_retrieve(self, query: str) -> List[Dict]:
        """向量检索"""
        # 实际实现调用向量数据库
        return [{"text": f"关于{query}的相关文档..."}]

    def _combine_contexts(self, graph_context: str, vector_context: str) -> str:
        """融合图谱和向量上下文"""
        combined = f"""
=== 知识图谱信息 ===
{graph_context}

=== 文档检索信息 ===
{vector_context}
"""
        return combined

    def _generate_answer(self, query: str, context: str, subgraph) -> str:
        """生成答案"""
        prompt = f"""
基于以下知识图谱和文档信息，回答问题。

问题：{query}

信息：
{context}

请提供准确、详细的答案。如果信息不足，请明确说明。
"""

        return self.llm.generate(prompt)


# 使用示例
if __name__ == "__main__":
    # 创建组件
    kg = KnowledgeGraph()
    # ... 添加实体和关系 ...

    vector_store = None  # 实际使用时连接向量数据库
    llm_client = None  # 实际使用时连接LLM

    # 创建GraphRAG系统
    graph_rag = GraphRAGSystem(kg, vector_store, llm_client)

    # 查询
    result = graph_rag.query("马斯克的公司最近收购了什么？")
    print(f"答案: {result['answer']}")
```

---

## 15.4 高级图检索

### 15.4.1 图嵌入

```python
# 文件名：graph_embedding.py
"""
图嵌入技术
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class GraphEmbedding:
    """
    图嵌入：将图结构转换为向量表示

    方法：
    1. Node2Vec: 基于随机游走
    2. GraphSAGE: 图神经网络
    3. 简化方法：聚合节点特征
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)

    def embed_subgraph(self, subgraph: 'KnowledgeGraph') -> np.ndarray:
        """
        将子图嵌入为向量

        方法：聚合节点及其邻居的文本描述
        """
        embeddings = []

        for node in subgraph.graph.nodes():
            # 获取节点属性
            node_data = subgraph.graph.nodes[node]

            # 构建节点描述
            node_desc = f"{node}"
            if 'type' in node_data:
                node_desc += f" is a {node_data['type']}"

            # 添加邻居信息
            neighbors = subgraph.get_neighbors(node)
            if neighbors:
                neighbor_descs = [f"connected to {n} via {r}" for n, r in neighbors[:3]]
                node_desc += f", {', '.join(neighbor_descs)}"

            # 嵌入
            embedding = self.model.encode([node_desc])[0]
            embeddings.append(embedding)

        # 聚合（平均或求和）
        graph_embedding = np.mean(embeddings, axis=0)

        return graph_embedding
```

---

## 15.5 完整项目：GraphRAG问答系统

### 15.5.1 项目架构

```
GraphRAG问答系统架构

┌─────────────────────────────────────────────┐
│                  用户查询                    │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  实体识别器            │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                       │
    ┌───┴────┐            ┌────┴────────┐
    │图谱检索│            │  向量检索   │
    └───┬────┘            └────┬────────┘
        │                       │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │      上下文融合       │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │      答案生成         │
        └───────────────────────┘
```

### 15.5.2 完整实现

```python
# 文件名：graphrag_qa_system.py
"""
完整的GraphRAG问答系统
"""

from typing import List, Dict, Tuple
import json


class GraphRAGQuestionAnswering:
    """
    GraphRAG问答系统

    支持多跳推理和关系查询
    """

    def __init__(self, kg_path: str, vector_store_path: str):
        # 加载知识图谱
        self.kg = self._load_kg(kg_path)

        # 加载向量存储
        self.vector_store = self._load_vector_store(vector_store_path)

        # 初始化LLM
        self.llm = self._init_llm()

    def _load_kg(self, path: str) -> 'KnowledgeGraph':
        """加载知识图谱"""
        kg = KnowledgeGraph()

        # 从文件加载（Neo4j CSV、JSON等）
        # 这里简化为构建示例
        # 实际使用时从数据库加载

        return kg

    def _load_vector_store(self, path: str):
        """加载向量存储"""
        # 实际实现加载ChromaDB等
        return None

    def _init_llm(self):
        """初始化LLM"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4")

    def answer(self, question: str) -> Dict:
        """
        回答问题

        Args:
            question: 用户问题

        Returns:
            {
                'answer': str,
                'reasoning_path': List[str],
                'confidence': float
            }
        """
        print(f"\n问题: {question}")

        # 步骤1：实体识别
        entities = self._extract_entities(question)
        print(f"实体: {entities}")

        # 步骤2：确定查询类型
        query_type = self._classify_query(question, entities)
        print(f"查询类型: {query_type}")

        # 步骤3：执行相应查询策略
        if query_type == "factual":
            result = self._factual_query(entities, question)
        elif query_type == "multi_hop":
            result = self._multi_hop_query(entities, question)
        else:
            result = self._vector_rag_query(question)

        return result

    def _extract_entities(self, question: str) -> List[str]:
        """实体识别（简化版）"""
        import re

        # 提取大写开头的词作为候选实体
        candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        return list(set(candidates))

    def _classify_query(self, question: str, entities: List[str]) -> str:
        """分类查询类型"""
        # 简化规则
        if len(entities) >= 2:
            return "multi_hop"
        elif any(word in question.lower() for word in ["谁", "什么", "哪个", "如何"]):
            return "factual"
        else:
            return "general"

    def _multi_hop_query(self, entities: List[str], question: str) -> Dict:
        """多跳查询"""
        reasoning_path = []

        # 寻找实体间的关系路径
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]

            # 查找路径
            path = self.kg.find_path(entity1, entity2)

            if path:
                reasoning_path = path
                # 基于路径生成答案
                answer = self._explain_path(path, question)
            else:
                answer = f"抱歉，找不到{entity1}和{entity2}之间的关系"
        else:
            # 单实体查询
            answer = self._single_entity_query(entities[0], question)

        return {
            'answer': answer,
            'reasoning_path': reasoning_path,
            'confidence': 0.8 if reasoning_path else 0.5
        }

    def _explain_path(self, path: List[str], question: str) -> str:
        """解释路径"""
        explanation = f"推理路径：{' → '.join(path)}\n\n"

        # 获取路径上的关系
        path_details = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.kg.graph.get_edge_data(u, v)
            relation = edge_data.get('relation', 'related')
            path_details.append(f"{u} --{relation}--> {v}")

        explanation += "路径详情：\n" + "\n".join(path_details)

        # 生成自然语言答案
        answer = self.llm.predict(f"""
基于以下路径信息，回答问题。

问题：{question}

推理路径：
{explanation}

请生成自然语言的答案。
""")

        return answer


# 使用示例
if __name__ == "__main__":
    # 创建系统
    qa_system = GraphRAGQuestionAnswering(
        kg_path="./data/knowledge_graph.json",
        vector_store_path="./data/vector_store"
    )

    # 测试查询
    questions = [
        "马斯克的火箭公司是什么？",
        "SpaceX最近一次发射是什么？",
        "特斯拉和SpaceX有什么关系？"
    ]

    for q in questions:
        result = qa_system.answer(q)
        print(f"\n答案: {result['answer']}")
        print(f"推理路径: {result.get('reasoning_path', 'N/A')}\n")
```

---

## 练习题

### 练习15.1：构建小型知识图谱（基础）

**题目**：为特定领域构建小型知识图谱

**要求**：
1. 选择一个领域（如技术栈、影视、体育等）
2. 定义实体类型和关系类型
3. 至少50个实体和100个关系
4. 实现图谱可视化

---

### 练习15.2：实现GraphRAG（进阶）

**题目**：实现完整的GraphRAG系统

**要求**：
1. 实体识别模块
2. 图谱检索模块
3. 子图嵌入
4. 与向量检索融合
5. 评估GraphRAG vs 向量RAG的效果

---

### 练习15.3：多跳问答系统（挑战）

**题目**：构建支持多跳推理的问答系统

**功能需求**：
1. 支持2-3跳推理
2. 可视化推理路径
3. 处理复杂问题
4. 提供可信度评分

---

## 总结

### 本章要点

1. **知识图谱优势**
   - 结构化知识表示
   - 关系推理能力
   - 多跳查询支持
   - 可解释性强

2. **GraphRAG流程**
   - 实体识别 → 图谱检索 → 子图嵌入 → 答案生成
   - 结合图谱和向量检索

3. **高级技术**
   - 图嵌入（Node2Vec、GraphSAGE）
   - 图神经网络
   - 图数据库（Neo4j）

### 学习检查清单

- [ ] 理解知识图谱的价值
- [ ] 掌握GraphRAG实现
- [ ] 能够构建小型知识图谱
- [ ] 实现多跳推理问答

### 下一步学习

- **下一章**：[第16章：多模态RAG](./16-多模态RAG.md)
- **相关资源**：
  - Neo4j: https://neo4j.com/
  - GraphRAG Paper: https://arxiv.org/abs/2404.16130

---

**恭喜完成第15章！** 🎉

**最后一章：多模态RAG！** → [第16章](./16-多模态RAG.md)
