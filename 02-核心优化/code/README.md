# InteliKB v2.0 代码仓库

> 完整的RAG系统优化实现代码

---

## 📁 项目结构

```
RAG完整教程/
├── 02-核心优化/
│   ├── code/                          # 模块2完整代码
│   │   ├── config.py                  # 配置文件
│   │   ├── models/                    # 模型相关
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py          # 嵌入模型封装
│   │   │   ├── reranker.py           # 重排序模型
│   │   │   └── llm.py                # LLM接口
│   │   ├── retrievers/                # 检索器
│   │   │   ├── __init__.py
│   │   │   ├── vector.py             # 向量检索
│   │   │   ├── bm25.py               # BM25检索
│   │   │   └── hybrid.py             # 混合检索
│   │   ├── cache/                     # 缓存系统
│   │   │   ├── __init__.py
│   │   │   ├── l1_cache.py           # L1内存缓存
│   │   │   └── l2_cache.py           # L2 Redis缓存
│   │   ├── optimization/              # 优化技术
│   │   │   ├── __init__.py
│   │   │   ├── query_enhancement.py  # 查询增强
│   │   │   ├── chunking.py           # 分块策略
│   │   │   └── advanced_rag.py       # 高级RAG模式
│   │   ├── engine/                    # RAG引擎
│   │   │   ├── __init__.py
│   │   │   └── rag_engine.py         # 主引擎
│   │   ├── evaluation/                # 评估工具
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py            # 评估指标
│   │   │   └── ab_testing.py         # A/B测试
│   │   ├── api/                       # API服务
│   │   │   ├── __init__.py
│   │   │   └── app.py                # FastAPI应用
│   │   ├── monitoring/                # 监控
│   │   │   ├── __init__.py
│   │   │   └── metrics_collector.py  # 指标收集
│   │   ├── utils/                     # 工具函数
│   │   │   ├── __init__.py
│   │   │   └── helpers.py            # 辅助函数
│   │   └── main.py                    # 主程序入口
```

---

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### requirements.txt

```
# 核心依赖
sentence-transformers==2.2.2
rank-bm25==0.2.2
chromadb==0.4.18
openai==1.3.5
numpy==1.24.3
pandas==2.0.3

# 可选依赖
redis==4.6.0          # L2缓存
fastapi==0.104.1      # API服务
uvicorn==0.24.0       # ASGI服务器
prometheus-client==0.19.0  # 监控

# 开发工具
jupyter==1.0.0
pytest==7.4.3
black==23.12.0
```

---

## 📦 核心模块说明

### 1. config.py - 配置管理

```python
"""
InteliKB v2.0 配置
"""

class Config:
    # 嵌入模型
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 768
    EMBEDDING_BATCH_SIZE = 32

    # 分块
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    CHUNKING_STRATEGY = "semantic"

    # 检索
    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
    RRF_K = 60
    INITIAL_TOP_K = 50
    FINAL_TOP_K = 10

    # 重排序
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # 缓存
    L1_CACHE_SIZE = 1000
    L1_CACHE_TTL = 3600
    L2_CACHE_TTL = 86400

    # LLM
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500

    # 并发
    MAX_WORKERS = 10
    BATCH_SIZE = 8
```

### 2. engine/rag_engine.py - 核心引擎

```python
"""
InteliKB v2.0 RAG引擎
整合所有优化技术
"""

from ..config import Config
from ..models.embedding import EmbeddingModel
from ..retrievers.hybrid import HybridRetriever
from ..models.reranker import Reranker
from ..cache.l1_cache import L1Cache
from ..cache.l2_cache import L2Cache
from ..evaluation.metrics import compute_metrics


class IntelikBEngine:
    """InteliKB v2.0 引擎"""

    def __init__(self, config: Config):
        self.config = config

        # 初始化组件
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
        self.retriever = HybridRetriever(config)
        self.reranker = Reranker(config.RERANKER_MODEL)
        self.l1_cache = L1Cache(config.L1_CACHE_SIZE, config.L1_CACHE_TTL)
        self.l2_cache = L2Cache(config.L2_CACHE_TTL)

    def add_documents(self, documents, metadata=None):
        """添加文档"""
        # 嵌入文档
        embeddings = self.embedding_model.encode(documents)

        # 添加到检索器
        self.retriever.add_documents(documents, embeddings, metadata)

    def query(self, query: str, use_cache: bool = True):
        """查询"""
        # 检查缓存
        if use_cache:
            cached = self._get_cached(query)
            if cached:
                return cached

        # 检索
        candidates = self.retriever.retrieve(query)

        # 重排序
        reranked = self.reranker.rerank(query, candidates)

        # 生成答案
        answer = self._generate_answer(query, reranked)

        # 缓存结果
        if use_cache:
            self._set_cached(query, answer)

        return answer

    def _get_cached(self, query: str):
        """获取缓存"""
        # L1
        result = self.l1_cache.get(query)
        if result:
            return result

        # L2
        result = self.l2_cache.get(query)
        if result:
            # 回填L1
            self.l1_cache.set(query, result)
            return result

        return None

    def _set_cached(self, query: str, value: dict):
        """设置缓存"""
        self.l1_cache.set(query, value)
        self.l2_cache.set(query, value)
```

### 3. api/app.py - API服务

```python
"""
FastAPI服务
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from ..engine.rag_engine import IntelikBEngine
from ..config import Config


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cache_hit: bool
    response_time_ms: float


app = FastAPI(title="InteliKB v2.0 API")

# 初始化引擎
config = Config()
engine = IntelikBEngine(config)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    import time
    start = time.time()

    try:
        result = engine.query(request.query, request.use_cache)
        response_time = (time.time() - start) * 1000

        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            cache_hit=result.get('cache', 'None') != 'None',
            response_time_ms=response_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return engine.get_stats()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🧪 测试

### 运行测试

```bash
# 单元测试
pytest tests/ -v

# 性能测试
python tests/performance_test.py

# A/B测试
python tests/ab_test.py
```

### 测试覆盖

- `tests/test_embedding.py` - 嵌入模型测试
- `tests/test_retriever.py` - 检索器测试
- `tests/test_cache.py` - 缓存测试
- `tests/test_engine.py` - 引擎集成测试

---

## 📊 性能基准

### 预期性能

| 指标 | v1.0 | v2.0 | 提升 |
|------|------|------|------|
| Hit Rate | 0.60 | 0.85 | +42% |
| MRR | 0.50 | 0.75 | +50% |
| P95延迟 | 3000ms | 1500ms | -50% |
| QPS | 5 | 50 | +900% |

### 运行基准测试

```bash
python scripts/run_benchmark.py
```

---

## 🐳 Docker部署

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 运行服务
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 部署命令

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

---

## 📈 监控

### Prometheus配置

```python
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
query_counter = Counter('rag_queries_total', 'Total queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

# 使用
query_counter.inc()
with query_duration.time():
    result = engine.query(query)
```

### Grafana Dashboard

导入 `monitoring/grafana_dashboard.json` 获得预配置的仪表板。

---

## 🔧 开发指南

### 代码规范

```bash
# 格式化代码
black .

# 运行linter
flake8 .

# 类型检查
mypy .
```

### 提交代码

```bash
# 运行测试
pytest

# 格式化
black .

# 提交
git add .
git commit -m "feat: add new feature"
```

---

## 📚 文档

- API文档：http://localhost:8000/docs
- 教程：`../docs/`
- 示例：`examples/`

---

## 🤝 贡献指南

1. Fork本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交Pull Request

---

## 📝 许可证

MIT License

---

## 🙏 致谢

- LlamaIndex团队
- LangChain团队
- Sentence-Transformers团队
- 所有贡献者

---

**最后更新**：2025-02-10
**版本**：v2.0.0
