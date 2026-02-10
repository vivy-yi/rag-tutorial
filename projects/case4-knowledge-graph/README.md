# 案例4：企业知识图谱问答系统

> 构建基于知识图谱的企业级问答系统，实现复杂关系推理和多跳查询

---

## 📋 案例概述

### 业务场景

大型企业面临知识管理挑战：
- ✗ 知识分散在各个系统
- ✗ 员工难以找到相关信息
- ✗ 知识关系复杂难理解
- ✗ 新员工培训周期长

### GraphRAG解决方案

构建企业知识图谱问答系统：
- ✅ 统一知识表示
- ✅ 复杂关系推理
- ✅ 可视化知识探索
- ✅ 智能问答交互

---

## 🎯 功能需求

### 核心功能

1. **知识图谱构建**
   - 从文档提取实体和关系
   - 多源知识融合
   - 图谱可视化
   - 动态更新

2. **智能问答**
   - 自然语言查询
   - 多跳推理
   - 关系路径展示
   - 答案解释

3. **知识探索**
   - 实体关系导航
   - 图谱浏览
   - 相关知识推荐
   - 知识依赖分析

4. **图谱管理**
   - 实体/关系增删改
   - 数据导入导出
   - 版本控制
   - 权限管理

---

## 🏗️ 系统架构

### 架构图

```
┌──────────────────────────────────────────┐
│         Web界面 (React/D3.js)            │
│  - 知识图谱可视化                         │
│  - 问答交互                               │
│  - 实体详情展示                           │
└──────────────┬───────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────┐
│          API服务 (FastAPI)               │
│  - 图谱查询接口                           │
│  - 问答处理                               │
│  - 数据管理                               │
└───────┬────────────────┬─────────────────┘
        │                │
    ┌───┴────┐      ┌───┴────────┐
    │图谱检索│      │  LLM生成   │
    │(Neo4j) │      │  (GPT-4)   │
    └────────┘      └────────────┘
```

### 技术栈

**图谱存储**：
- Neo4j (图数据库)
- NetworkX (图计算)

**NLP处理**：
- spaCy (实体识别)
- LangChain (图谱集成)
- OpenAI GPT-4

**后端**：
- FastAPI
- PyNeo4j (Neo4j驱动)

**前端**：
- React
- D3.js / Cytoscape.js (可视化)
- Ant Design

---

## 💻 核心实现

### 1. 知识图谱构建

```python
# graph_builder.py
from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import spacy

class EnterpriseKnowledgeGraph:
    """
    企业知识图谱构建器
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

        # 加载NER模型
        self.nlp = spacy.load("zh_core_web_sm")

    def close(self):
        """关闭连接"""
        self.driver.close()

    def create_constraints(self):
        """创建约束和索引"""
        with self.driver.session() as session:
            # 唯一约束
            session.run("""
                CREATE CONSTRAINT entity_id_unique
                IF NOT EXISTS FOR (e:Entity)
                REQUIRE e.id IS UNIQUE
            """)

            # 索引
            session.run("""
                CREATE INDEX entity_name_index
                IF NOT EXISTS FOR (e:Entity)
                ON (e.name)
            """)

    def add_entity(self,
                   entity_id: str,
                   name: str,
                   entity_type: str,
                   properties: Dict = None):
        """
        添加实体

        Args:
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 其他属性
        """
        with self.driver.session() as session:
            query = """
                MERGE (e:Entity:$type {id: $id})
                SET e.name = $name,
                    e.updated_at = datetime()
            """

            params = {
                "id": entity_id,
                "type": entity_type,
                "name": name
            }

            if properties:
                for key, value in properties.items():
                    query += f", e.{key} = ${key}"
                    params[key] = value

            session.run(query, params)

    def add_relation(self,
                    source_id: str,
                    target_id: str,
                    relation_type: str,
                    properties: Dict = None):
        """
        添加关系

        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            properties: 关系属性
        """
        with self.driver.session() as session:
            query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:$type]->(target)
                SET r.updated_at = datetime()
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "type": relation_type
            }

            if properties:
                for key, value in properties.items():
                    query += f", r.{key} = ${key}"
                    params[key] = value

            session.run(query, params)

    def extract_from_document(self,
                            document: str,
                            doc_id: str) -> Tuple[int, int]:
        """
        从文档中提取知识

        Args:
            document: 文档文本
            doc_id: 文档ID

        Returns:
            (实体数, 关系数)
        """
        # NLP处理
        doc = self.nlp(document)

        # 提取实体（简化版）
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT", "TECH"]:
                entity_id = f"{doc_id}_{ent.start_char}"

                self.add_entity(
                    entity_id=entity_id,
                    name=ent.text,
                    entity_type=ent.label_,
                    properties={
                        "source_doc": doc_id,
                        "confidence": 0.9
                    }
                )

                entities.append({
                    "id": entity_id,
                    "text": ent.text,
                    "label": ent.label_
                })

        # 提取关系（基于依存关系）
        relations = []
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                # 简化：假设相邻实体有关系
                head_text = token.head.text
                dep_text = token.text

                # 查找对应的实体
                head_entity = next(
                    (e for e in entities if e["text"] == head_text),
                    None
                )
                dep_entity = next(
                    (e for e in entities if e["text"] == dep_text),
                    None
                )

                if head_entity and dep_entity:
                    self.add_relation(
                        source_id=head_entity["id"],
                        target_id=dep_entity["id"],
                        relation_type=token.dep_,
                        properties={"source_doc": doc_id}
                    )
                    relations.append(1)

        return len(entities), len(relations)

    def import_from_json(self, json_file: str):
        """
        从JSON导入知识图谱

        JSON格式:
        {
            "entities": [
                {"id": "E1", "name": "张三", "type": "Person"},
                ...
            ],
            "relations": [
                {"source": "E1", "target": "E2", "type": "工作于"},
                ...
            ]
        }
        """
        import json

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 添加实体
        for entity in data.get("entities", []):
            self.add_entity(
                entity_id=entity["id"],
                name=entity["name"],
                entity_type=entity["type"],
                properties=entity.get("properties", {})
            )

        # 添加关系
        for relation in data.get("relations", []):
            self.add_relation(
                source_id=relation["source"],
                target_id=relation["target"],
                relation_type=relation["type"],
                properties=relation.get("properties", {})
            )
```

### 2. GraphRAG查询

```python
# graph_rag_query.py
from typing import List, Dict, Optional
from neo4j import GraphDatabase

class GraphRAGQuery:
    """
    基于知识图谱的RAG查询
    """

    def __init__(self, kg: EnterpriseKnowledgeGraph):
        """
        初始化

        Args:
            kg: 知识图谱实例
        """
        self.kg = kg

    def query(self,
             question: str,
             max_hops: int = 3) -> Dict:
        """
        执行查询

        Args:
            question: 自然语言问题
            max_hops: 最大跳数

        Returns:
            查询结果
        """
        # 1. 识别实体
        entities = self._extract_entities(question)

        if not entities:
            return {
                "answer": "抱歉，我无法识别问题中的实体",
                "paths": []
            }

        # 2. 图谱检索
        paths = self._graph_retrieve(entities, max_hops)

        # 3. 构建上下文
        context = self._build_context(paths)

        # 4. 生成答案
        answer = self._generate_answer(question, context, paths)

        return {
            "answer": answer,
            "entities": entities,
            "paths": paths,
            "context": context
        }

    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体ID"""
        # 使用NER提取实体名称
        doc = self.kg.nlp(text)
        entity_names = [ent.text for ent in doc.ents]

        # 在图谱中查找
        entity_ids = []
        with self.kg.driver.session() as session:
            for name in entity_names:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $name
                    RETURN e.id as id
                    LIMIT 1
                """, {"name": name})

                record = result.single()
                if record:
                    entity_ids.append(record["id"])

        return entity_ids

    def _graph_retrieve(self,
                       entity_ids: List[str],
                       max_hops: int) -> List[Dict]:
        """
        图谱检索：多跳路径查找

        Args:
            entity_ids: 起始实体ID列表
            max_hops: 最大跳数

        Returns:
            路径列表
        """
        paths = []

        with self.kg.driver.session() as session:
            for entity_id in entity_ids:
                # 查找所有路径
                query = """
                    MATCH path = (start:Entity {id: $entity_id})-[*1..{max_hops}]-(end:Entity)
                    RETURN [node in nodes(path) | {
                        id: node.id,
                        name: node.name,
                        type: labels(node)[0]
                    }] as nodes,
                    [rel in relationships(path) | {
                        type: type(rel),
                        source: startNode(rel).id,
                        target: endNode(rel).id
                    }] as rels
                    ORDER BY length(path)
                    LIMIT 10
                """

                result = session.run(
                    query,
                    {"entity_id": entity_id, "max_hops": max_hops}
                )

                for record in result:
                    paths.append({
                        "nodes": record["nodes"],
                        "rels": record["rels"]
                    })

        return paths

    def _build_context(self, paths: List[Dict]) -> str:
        """构建上下文文本"""
        if not paths:
            return "未找到相关知识"

        context_parts = []

        for i, path in enumerate(paths[:5], 1):
            # 构建路径描述
            nodes = path["nodes"]
            path_str = " → ".join([n["name"] for n in nodes])

            context_parts.append(f"路径{i}: {path_str}")

        return "\n".join(context_parts)

    def _generate_answer(self,
                        question: str,
                        context: str,
                        paths: List[Dict]) -> str:
        """
        生成答案

        这里可以使用LLM，也可以用规则
        """
        # 规则生成（简化版）
        answer_parts = [f"基于知识图谱，找到以下信息：\n"]
        answer_parts.append(context)

        if paths:
            answer_parts.append("\n主要发现：")
            for i, path in enumerate(paths[:3], 1):
                nodes = path["nodes"]
                answer_parts.append(f"{i}. {nodes[0]['name']}到{nodes[-1]['name']}的关系路径")

        return "\n".join(answer_parts)
```

### 3. 可视化接口

```python
# api/visualization.py
from fastapi import FastAPI, Query
from typing import List, Dict
from neo4j import GraphDatabase

app = FastAPI(title="知识图谱可视化API")

@app.get("/api/graph/subgraph")
async def get_subgraph(
    entity_id: str = Query(..., description="中心实体ID"),
    hops: int = Query(2, description="跳数")
):
    """
    获取子图用于可视化

    Returns:
        节点和边的列表
    """
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

    try:
        with driver.session() as session:
            # 查询子图
            query = """
                MATCH path = (start:Entity {id: $entity_id})-[*1..$hops]-(end:Entity)
                WITH nodes(path) as nodes,
                     relationships(path) as rels
                UNWIND nodes as node
                WITH collect(DISTINCT node) as all_nodes,
                     rels
                UNWIND rels as rel
                RETURN all_nodes as nodes,
                       collect(DISTINCT rel) as relationships
            """

            result = session.run(
                query,
                {"entity_id": entity_id, "hops": hops}
            )

            record = result.single()

            # 格式化节点
            nodes = []
            for node in record["nodes"]:
                nodes.append({
                    "id": node["id"],
                    "label": node["name"],
                    "type": list(node.labels())[0]
                })

            # 格式化边
            edges = []
            for rel in record["relationships"]:
                edges.append({
                    "source": rel.start_node["id"],
                    "target": rel.end_node["id"],
                    "label": type(rel).__name__
                })

            return {
                "nodes": nodes,
                "edges": edges
            }

    finally:
        driver.close()

@app.get("/api/graph/entity/{entity_id}")
async def get_entity_details(entity_id: str):
    """获取实体详情"""
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {id: $entity_id})
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                RETURN e as entity,
                       collect(DISTINCT {
                           id: related.id,
                           name: related.name,
                           type: labels(related)[0],
                           relation: type(r)
                       }) as relations
            """, {"entity_id": entity_id})

            record = result.single()

            entity = record["entity"]
            relations = record["relations"]

            return {
                "id": entity["id"],
                "name": entity["name"],
                "type": list(entity.labels())[0],
                "properties": dict(entity),
                "relations": relations
            }

    finally:
        driver.close()
```

### 4. React前端组件

```javascript
// frontend/src/components/GraphVisualization.jsx
import React, { useEffect, useState } from 'react';
import cytoscape from 'cytoscape';
import CytoscapeComponent from 'react-cytoscapejs';

const GraphVisualization = ({ entityId }) => {
  const [elements, setElements] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSubgraph();
  }, [entityId]);

  const fetchSubgraph = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/graph/subgraph?entity_id=${entityId}&hops=2`
      );
      const data = await response.json();

      // 转换为Cytoscape格式
      const elements = [
        ...data.nodes.map(node => ({
          data: {
            id: node.id,
            label: node.label,
            type: node.type
          }
        })),
        ...data.edges.map(edge => ({
          data: {
            id: `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            label: edge.label
          }
        }))
      ];

      setElements(elements);
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  const layout = {
    name: 'cose',
    animate: true,
    nodeDimensionsIncludeLabels: true
  };

  const stylesheet = [
    {
      selector: 'node',
      style: {
        'background-color': '#666',
        'label': 'data(label)',
        'width': 30,
        'height': 30
      }
    },
    {
      selector: 'node[type="Person"]',
      style: {
        'background-color': '#FF6B6B'
      }
    },
    {
      selector: 'node[type="Organization"]',
      style: {
        'background-color': '#4ECDC4'
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#ccc',
        'target-arrow-color': '#ccc',
        'target-arrow-shape': 'triangle'
      }
    }
  ];

  return (
    <div style={{ height: '600px' }}>
      <CytoscapeComponent
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        zoomingEnabled={true}
        maxZoom={3}
        minZoom={0.1}
        autounselectify={true}
        layout={layout}
        stylesheet={stylesheet}
      />
    </div>
  );
};

export default GraphVisualization;
```

---

## 📊 数据示例

### 企业知识图谱数据结构

```json
{
  "entities": [
    {
      "id": "E001",
      "name": "张三",
      "type": "Person",
      "properties": {
        "department": "技术部",
        "position": "高级工程师",
        "email": "zhangsan@company.com"
      }
    },
    {
      "id": "E002",
      "name": "李四",
      "type": "Person",
      "properties": {
        "department": "产品部",
        "position": "产品经理",
        "email": "lisi@company.com"
      }
    },
    {
      "id": "E003",
      "name": "RAG系统",
      "type": "Project",
      "properties": {
        "status": "进行中",
        "priority": "高"
      }
    },
    {
      "id": "E004",
      "name": "技术部",
      "type": "Department",
      "properties": {
        "location": "3楼",
        "head": "王五"
      }
    }
  ],
  "relations": [
    {
      "source": "E001",
      "target": "E003",
      "type": "负责",
      "properties": {
        "role": "技术负责人",
        "since": "2024-01"
      }
    },
    {
      "source": "E002",
      "target": "E003",
      "type": "管理",
      "properties": {
        "role": "产品经理"
      }
    },
    {
      "source": "E001",
      "target": "E004",
      "type": "隶属",
      "properties": {}
    },
    {
      "source": "E002",
      "target": "E004",
      "type": "协作",
      "properties": {
        "frequency": "每周会议"
      }
    }
  ]
}
```

---

## 🧪 测试场景

### 场景1：员工信息查询

**问题**："张三负责哪些项目？"

**执行流程**：
1. 识别实体：张三
2. 图谱检索：
   ```
   张三 -[负责]-> RAG系统
   张三 -[负责]-> 知识图谱系统
   ```
3. 答案生成

### 场景2：关系推理

**问题**："李四和张三有合作吗？"

**执行流程**：
1. 识别实体：李四、张三
2. 多跳检索：
   ```
   李四 -[管理]-> RAG系统 <-[负责]- 张三
   ```
3. 推理：两人共同参与RAG系统项目

### 场景3：组织分析

**问题**："技术部有哪些人？"

**执行流程**：
1. 识别实体：技术部
2. 图谱检索所有隶属关系
3. 返回所有技术人员

---

## 🎨 可视化示例

### 图谱展示

使用D3.js或Cytoscape.js展示：

```javascript
// 节点样式配置
const nodeStyles = {
  'Person': { color: '#FF6B6B', icon: 'user' },
  'Organization': { color: '#4ECDC4', icon: 'building' },
  'Project': { color: '#45B7D1', icon: 'folder' },
  'Technology': { color: '#96CEB4', icon: 'code' }
};

// 力导向布局
const layout = {
  type: 'force',
  nodeSpacing: 100,
  linkDistance: 150
};

// 交互功能
const interactions = {
  zoom: true,
  drag: true,
  hover: true,
  click: (node) => {
    showEntityDetails(node.id);
  }
};
```

---

## 📈 性能优化

### 1. 图谱索引

```python
# 创建索引
def create_indexes():
    indexes = [
        "CREATE INDEX ON :Entity(name)",
        "CREATE INDEX ON :Entity(type)",
        "CREATE INDEX ON :Entity(updated_at)"
    ]

    for index in indexes:
        session.run(index)
```

### 2. 查询优化

```python
# 使用参数化查询
def optimized_query(entity_ids: List[str]):
    query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e
    """

    session.run(query, entity_ids=entity_ids)
```

### 3. 缓存层

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_entity(entity_id: str):
    """缓存实体查询"""
    return fetch_entity_from_neo4j(entity_id)
```

---

## 🚀 部署指南

### Docker Compose

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  neo4j_data:
```

---

## 🎓 学习要点

完成本案例后，你将掌握：

### ✅ 知识图谱
- Neo4j图数据库
- 图谱建模方法
- Cypher查询语言
- 图谱可视化

### ✅ GraphRAG
- 实体识别和抽取
- 多跳推理算法
- 图谱检索优化
- 上下文构建

### ✅ 系统设计
- 图数据库架构
- API设计
- 前后端分离
- 性能优化

---

## 🚀 进阶方向

1. **高级功能**
   - 实体消歧
   - 关系推理
   - 时序图谱
   - 图嵌入

2. **企业特性**
   - 权限控制
   - 数据加密
   - 审计日志
   - 备份恢复

3. **智能增强**
   - 自动图谱更新
   - 知识推荐
   - 异常检测
   - 趋势分析

---

## 📚 参考资源

- [Neo4j文档](https://neo4j.com/docs/)
- [GraphRAG论文](https://arxiv.org/abs/2404.16130)
- [spaCy NER](https://spacy.io/usage/linguistic-features/#named-entities)

---

**开始构建你的企业知识图谱系统吧！** 🚀
