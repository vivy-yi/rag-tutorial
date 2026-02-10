# 案例6：企业级RAG平台

> 构建生产级、可扩展的企业RAG平台，包含完整的监控、部署和运维方案

---

## 📋 案例概述

### 业务场景

大型企业知识管理需求：
- ✗ 多源数据分散（文档、数据库、API）
- ✗ 高并发访问需求
- ✗ 严格的权限控制
- ✗ 7x24高可用要求
- ✗ 完善的监控和运维

### 企业级RAG平台

构建完整的企业RAG平台：
- ✅ 多数据源接入
- ✅ 微服务架构
- ✅ 分布式部署
- ✅ 完整监控体系
- ✅ CI/CD流程
- ✅ 安全合规

---

## 🎯 系统需求

### 功能需求

1. **多源数据接入**
   - 文档上传（PDF, Word, Markdown）
   - 数据库同步（PostgreSQL, MySQL）
   - API集成
   - 网页爬取
   - 实时数据流

2. **智能问答服务**
   - 多租户支持
   - 权限控制
   - 上下文管理
   - 多轮对话
   - API服务

3. **管理后台**
   - 数据源管理
   - 知识库管理
   - 用户管理
   - 访问控制
   - 使用分析

4. **监控运维**
   - 性能监控
   - 日志分析
   - 告警通知
   - 容量规划
   - 故障排查

### 非功能需求

- **性能**：P99延迟 < 2秒，支持1000+ QPS
- **可用性**：99.9% SLA
- **并发**：支持10000+ 并发用户
- **扩展性**：水平扩展能力
- **安全**：RBAC、数据加密、审计日志

---

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────┐
│                   负载均衡                        │
│                  (Nginx/ALB)                     │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴─────────┐
       │                 │
┌──────┴─────┐    ┌─────┴──────┐
│   前端服务   │    │  API网关   │
│  (React)    │    │ (Kong)    │
└──────┬──────┘    └─────┬──────┘
       │                │
       │    ┌───────────┴───────────┐
       │    │                       │
┌──────┴────┴──┐        ┌────────┴─────────┐
│  应用服务层  │        │   认证服务       │
│  (FastAPI)  │        │   (Keycloak)    │
└──────┬──────┘        └──────────────────┘
       │
┌──────┴──────────────────────────────┐
│            业务服务层                │
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │RAG服务  │ │索引服务  │ │用户服务││
│  └─────────┘ └─────────┘ └────────┘│
└──────────────────────────────────────┘
       │
┌──────┴──────────────────────────────┐
│            数据层                    │
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │向量DB   │ │PostgreSQL│ │  Redis ││
│  │(Pinecone)│ │         │ │        ││
│  └─────────┘ └─────────┘ └────────┘│
└──────────────────────────────────────┘
```

### 技术栈

**后端服务**：
- FastAPI (应用层)
- Kong (API网关)
- Keycloak (认证)

**数据存储**：
- Pinecone (向量数据库)
- PostgreSQL (关系数据库)
- Redis (缓存)
- RabbitMQ (消息队列)

**监控运维**：
- Prometheus (指标收集)
- Grafana (可视化)
- ELK Stack (日志)
- Jaeger (链路追踪)

**部署**：
- Kubernetes (容器编排)
- Docker (容器化)
- Helm (包管理)
- ArgoCD (GitOps)

**CI/CD**：
- GitHub Actions
- Docker Registry
- Kubernetes Rollouts

---

## 💻 核心实现

### 1. 微服务架构

```python
# services/rag_service.py
from fastapi import FastAPI, Depends, HTTPException
from typing import List, Optional
import logging

from core.config import settings
from core.security import get_current_user
from models.query import QueryRequest, QueryResponse
from services.retriever import HybridRetriever
from services.generator import AnswerGenerator
from services.cache import CacheService
from services.monitor import monitor_query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Service",
    version="1.0.0",
    description="Enterprise RAG Service"
)

# 初始化服务
retriever = HybridRetriever(settings)
generator = AnswerGenerator(settings)
cache = CacheService(settings)

@app.post("/api/v1/query", response_model=QueryResponse)
@monitor_query
async def query(
    request: QueryRequest,
    current_user = Depends(get_current_user)
):
    """
    RAG查询接口

    Args:
        request: 查询请求
        current_user: 当前用户（从JWT获取）

    Returns:
        查询响应
    """
    try:
        # 1. 检查缓存
        cached_result = await cache.get(
            current_user.tenant_id,
            request.query
        )

        if cached_result:
            logger.info(f"Cache hit for query: {request.query[:50]}")
            return cached_result

        # 2. 检索
        documents = await retriever.retrieve(
            query=request.query,
            tenant_id=current_user.tenant_id,
            top_k=request.top_k or 10,
            filters=request.filters or {}
        )

        # 3. 生成答案
        answer = await generator.generate(
            query=request.query,
            documents=documents,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )

        response = QueryResponse(
            answer=answer.text,
            sources=answer.sources,
            confidence=answer.confidence,
            latency_ms=answer.latency_ms
        )

        # 4. 缓存结果
        await cache.set(
            current_user.tenant_id,
            request.query,
            response,
            ttl=3600
        )

        return response

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "rag-service",
        "version": "1.0.0"
    }

@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus指标"""
    from prometheus_client import generate_latest, REGISTRY
    return generate_latest(REGISTRY)
```

### 2. 混合检索服务

```python
# services/hybrid_retriever.py
from typing import List, Dict, Optional
import asyncio
from sentence_transformers import CrossEncoder

from core.vector_store import VectorStore
from core.bm25 import BM25Retriever
from core.reranker import Reranker
from models.document import Document

class HybridRetriever:
    """
    混合检索器

    结合向量检索、BM25和重排序
    """

    def __init__(self, settings):
        self.vector_store = VectorStore(settings)
        self.bm25 = BM25Retriever(settings)
        self.reranker = Reranker(settings)

    async def retrieve(self,
                      query: str,
                      tenant_id: str,
                      top_k: int = 10,
                      filters: Dict = None) -> List[Document]:
        """
        混合检索

        Args:
            query: 查询文本
            tenant_id: 租户ID
            top_k: 返回结果数
            filters: 过滤条件

        Returns:
            检索文档列表
        """
        # 并行检索
        vector_task = self.vector_store.search(
            query, tenant_id, top_k=top_k*2
        )
        bm25_task = self.bm25.search(
            query, tenant_id, top_k=top_k*2
        )

        # 等待两个检索完成
        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )

        # RRF融合
        fused = self._rrf_fusion(
            [vector_results, bm25_results],
            k=60
        )

        # 应用过滤器
        if filters:
            fused = self._apply_filters(fused, filters)

        # 重排序
        reranked = await self.reranker.rerank(
            query, fused[:top_k*2]
        )

        return reranked[:top_k]

    def _rrf_fusion(self,
                   rankings: List[List[Document]],
                   k: int = 60) -> List[Document]:
        """
        RRF融合

        Args:
            rankings: 多个排序列表
            k: RRF参数

        Returns:
            融合后的文档列表
        """
        rrf_scores = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                if doc.id not in rrf_scores:
                    rrf_scores[doc.id] = {
                        'doc': doc,
                        'score': 0.0
                    }

                # RRF公式: 1 / (k + rank)
                rrf_scores[doc.id]['score'] += 1.0 / (k + rank)

        # 按分数排序
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [item['doc'] for item in sorted_docs]

    def _apply_filters(self,
                      documents: List[Document],
                      filters: Dict) -> List[Document]:
        """应用过滤器"""
        filtered = documents

        if 'category' in filters:
            filtered = [
                d for d in filtered
                if d.metadata.get('category') == filters['category']
            ]

        if 'date_from' in filters:
            filtered = [
                d for d in filtered
                if d.metadata.get('created_at') >= filters['date_from']
            ]

        return filtered
```

### 3. 监控服务

```python
# services/monitor.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# 定义指标
query_counter = Counter(
    'rag_queries_total',
    'Total queries',
    ['tenant_id', 'status']
)

query_duration = Histogram(
    'rag_query_duration_seconds',
    'Query duration',
    ['tenant_id']
)

cache_hits = Counter(
    'rag_cache_hits_total',
    'Total cache hits',
    ['tenant_id']
)

active_users = Gauge(
    'rag_active_users',
    'Active users',
    ['tenant_id']
)

def monitor_query(func):
    """查询监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        tenant_id = kwargs.get('tenant_id', 'unknown')

        try:
            result = await func(*args, **kwargs)

            # 记录成功查询
            query_counter.labels(
                tenant_id=tenant_id,
                status='success'
            ).inc()

            # 记录查询时间
            duration = time.time() - start
            query_duration.labels(
                tenant_id=tenant_id
            ).observe(duration)

            return result

        except Exception as e:
            # 记录失败查询
            query_counter.labels(
                tenant_id=tenant_id,
                status='error'
            ).inc()
            raise

    return wrapper

class MetricsCollector:
    """指标收集器"""

    @staticmethod
    def record_cache_hit(tenant_id: str):
        """记录缓存命中"""
        cache_hits.labels(tenant_id=tenant_id).inc()

    @staticmethod
    def update_active_users(tenant_id: str, count: int):
        """更新活跃用户数"""
        active_users.labels(tenant_id=tenant_id).set(count)

    @staticmethod
    def record_index_metrics(tenant_id: str,
                            total_docs: int,
                            index_size_mb: float):
        """记录索引指标"""
        index_size = Gauge(
            'rag_index_size_mb',
            'Index size in MB',
            ['tenant_id']
        )
        index_size.labels(tenant_id=tenant_id).set(index_size_mb)

        total_docs_gauge = Gauge(
            'rag_total_documents',
            'Total documents',
            ['tenant_id']
        )
        total_docs_gauge.labels(tenant_id=tenant_id).set(total_docs)
```

### 4. Kubernetes部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
        version: v1.0.0
    spec:
      containers:
      - name: rag-service
        image: registry.example.com/rag-service:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api_key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: production
spec:
  selector:
    app: rag-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 5. CI/CD流程

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  REGISTRY: registry.example.com
  IMAGE_NAME: rag-service

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.REGISTRY_USER }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - name: Build and Push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

      - name: Run Tests
        run: |
          docker-compose -f docker-compose.test.yml up
          docker-compose -f docker-compose.test.yml run pytest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/rag-service \
            rag-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production

      - name: Verify Deployment
        run: |
          kubectl rollout status deployment/rag-service -n production

      - name: Notify Slack
        uses: 8398a7/action-slack-send@v3
        with:
          status: ${{ job.status }}
          text: |
            Deployment to production completed!
            Commit: ${{ github.sha }}
            Author: ${{ github.actor }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 6. 监控Dashboard

```python
# monitoring/grafana_dashboard.json
{
  "dashboard": {
    "title": "RAG Service Dashboard",
    "panels": [
      {
        "title": "Query Rate (QPS)",
        "targets": [
          {
            "expr": "rate(rag_queries_total[5m])",
            "legendFormat": "{{tenant_id}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Query Duration (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_query_duration_seconds)",
            "legendFormat": "{{tenant_id}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / rate(rag_queries_total[5m])",
            "legendFormat": "{{tenant_id}}"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Active Users",
        "targets": [
          {
            "expr": "rag_active_users",
            "legendFormat": "{{tenant_id}}"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(rag_queries_total{status=\"error\"}[5m]) / rate(rag_queries_total[5m])"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

---

## 📊 运维最佳实践

### 1. 健康检查

```python
# health/health_check.py
from fastapi import FastAPI
from core.database import db
from core.cache import cache

app = FastAPI()

@app.get("/health/live")
async def liveness():
    """存活检查"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """就绪检查"""
    checks = {
        "database": await db.check_connection(),
        "cache": await cache.check_connection(),
        "vector_store": await vector_store.check_connection()
    }

    is_ready = all(checks.values())

    status_code = 200 if is_ready else 503
    return JSONResponse(
        content=checks,
        status_code=status_code
    )
```

### 2. 日志规范

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging():
    """配置日志"""
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    )

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # 结构化日志
    logger.info("service_started", extra={
        "service": "rag-service",
        "version": "1.0.0"
    })
```

### 3. 配置管理

```python
# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """配置管理"""

    # 应用配置
    app_name: str = "rag-service"
    app_version: str = "1.0.0"
    debug: bool = False

    # 数据库
    database_url: str
    redis_url: str

    # 向量数据库
    pinecone_api_key: str
    pinecone_environment: str

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    # 监控
    enable_metrics: bool = True
    enable_tracing: bool = True

    # 安全
    secret_key: str
    jwt_algorithm: str = "HS256"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 🧪 测试策略

### 1. 单元测试

```python
# tests/test_retriever.py
import pytest
from services.hybrid_retriever import HybridRetriever

@pytest.fixture
async def retriever():
    return HybridRetriever(settings)

@pytest.mark.asyncio
async def test_hybrid_retrieve(retriever):
    """测试混合检索"""
    documents = await retriever.retrieve(
        query="什么是RAG？",
        tenant_id="test_tenant",
        top_k=5
    )

    assert len(documents) == 5
    assert all(hasattr(doc, 'id') for doc in documents)
```

### 2. 集成测试

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query_api():
    """测试查询API"""
    response = client.post(
        "/api/v1/query",
        json={
            "query": "什么是RAG？",
            "top_k": 5
        },
        headers={"Authorization": "Bearer token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
```

### 3. 性能测试

```python
# tests/performance/load_test.py
import asyncio
import time
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    """性能测试用户"""
    wait_time = between(1, 3)

    @task
    def query(self):
        response = self.client.post(
            "/api/v1/query",
            json={
                "query": "测试查询",
                "top_k": 5
            },
            headers={"Authorization": f"Bearer {self.token}"}
        )

        assert response.status_code == 200
```

---

## 📈 容量规划

### 扩展策略

**垂直扩展**：
- 增加Pod资源（CPU, Memory）
- 适用于：单服务性能瓶颈

**水平扩展**：
- 增加Pod副本数
- 通过HPA自动扩展
- 适用于：高并发场景

**数据库扩展**：
- 读写分离
- 分库分表
- 连接池优化

### 性能基准

| 指标 | 目标值 | 监控方式 |
|------|--------|----------|
| P50延迟 | < 500ms | Prometheus histogram |
| P95延迟 | < 2s | Prometheus histogram |
| P99延迟 | < 5s | Prometheus histogram |
| QPS | 1000+ | Prometheus rate |
| 并发用户 | 10000+ | Application metrics |
| 可用性 | 99.9% | Uptime monitor |

---

## 🎓 学习要点

完成本案例后，你将掌握：

### ✅ 微服务架构
- 服务拆分
- API网关
- 服务发现
- 负载均衡

### ✅ 企业级部署
- Kubernetes
- Docker容器化
- Helm Charts
- 滚动更新

### ✅ 监控运维
- Prometheus监控
- Grafana可视化
- ELK日志分析
- Jaeger链路追踪

### ✅ CI/CD
- GitHub Actions
- 自动化测试
- 自动部署
- GitOps

---

## 🚀 进阶方向

1. **高可用**
   - 多区域部署
   - 灾难恢复
   - 故障自愈
   - 蓝绿部署

2. **性能优化**
   - 缓存策略
   - 查询优化
   - 模型量化
   - 边缘计算

3. **安全加固**
   - 零信任网络
   - 数据加密
   - 安全审计
   - 渗透测试

---

## 📚 参考资源

- [Kubernetes文档](https://kubernetes.io/docs/)
- [Prometheus实践](https://prometheus.io/docs/)
- [FastAPI性能](https://fastapi.tiangolo.com/benchmarks/)

---

**恭喜完成所有案例！你已具备构建企业级RAG平台的能力！** 🚀
