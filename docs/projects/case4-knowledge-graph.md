# 案例4：企业知识图谱问答

> **难度**: ⭐⭐⭐ 高级 | **技术栈**: GraphRAG, Neo4j, NetworkX, 多跳推理

结合知识图谱的RAG系统，支持复杂的多跳推理和关系查询

---

## 🎯 案例特点

- ✅ **GraphRAG**: 图谱+向量混合检索
- ✅ **多跳推理**: 复杂关系链推理
- ✅ **路径可视化**: 推理路径可视化展示
- ✅ **Neo4j集成**: 专业图数据库

---

## 🚀 快速开始

```bash
cd projects/case4-knowledge-graph
pip install -r requirements.txt
# 启动Neo4j docker
docker-compose up -d
python main.py
```

---

## 🔑 核心技术

### 知识图谱构建

```python
from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687"):
        self.driver = GraphDatabase.driver(uri)

    def create_entity(self, entity_type, name, properties):
        """创建实体节点"""
        query = """
        CREATE (e:{type} $props)
        RETURN e
        """.format(type=entity_type)
        with self.driver.session() as session:
            return session.run(query, props=properties)

    def create_relation(self, from_entity, to_entity, relation_type):
        """创建关系"""
        query = """
        MATCH (a), (b)
        WHERE a.name = $from AND b.name = $to
        CREATE (a)-[r:{rel}]->(b)
        RETURN r
        """.format(rel=relation_type)
        with self.driver.session() as session:
            return session.run(query, from=from_entity, to=to)
```

### 多跳推理

```python
def multi_hop_query(query_entity, relation_path, max_hops=3):
    """多跳推理查询"""
    # 构建Cypher查询
    cypher = """
    MATCH path = (start {name: $entity})
    """

    # 添加跳数
    for i in range(max_hops):
        cypher += f"-[r{i}]->(n{i}) "

    cypher += "RETURN path"

    # 执行查询
    result = graph_db.run(cypher, entity=query_entity)
    return visualize_paths(result)
```

---

## 📊 图谱示例

```
(产品A) -[属于]-> (类别X) -[相关于]-> (技术Y)
    ↓
  [配件]
    ↓
(供应商Z) -[位于]-> (城市P)
```

**查询**: "产品A的供应商在哪个城市？"

**推理路径**: 产品A → 配件 → 供应商 → 城市

---

## 🎓 学习要点

1. **知识图谱**
   - 实体和关系建模
   - 图数据库操作
   - Cypher查询语言

2. **图算法**
   - 最短路径
   - 子图匹配
   - 关联挖掘

3. **GraphRAG**
   - 图谱+向量融合
   - 结构化+非结构化
   - 推理优化

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case4-knowledge-graph)**

**[← 返回案例列表](index.md)**
