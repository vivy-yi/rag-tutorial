# 案例6：企业级RAG平台

> **难度**: ⭐⭐⭐⭐ 专家 | **技术栈**: FastAPI, JWT, Redis, PostgreSQL, Docker, K8s

生产级RAG平台，包含完整的API、认证、缓存、监控和部署

---

## 🎯 案例特点

- ✅ **RESTful API**: FastAPI高性能接口
- ✅ **JWT认证**: 完整的用户权限系统
- ✅ **Redis缓存**: 查询结果缓存优化
- ✅ **异步处理**: Celery任务队列
- ✅ **监控告警**: Prometheus + Grafana
- ✅ **容器部署**: Docker + Kubernetes

---

## 🚀 快速开始

```bash
cd projects/case6-enterprise-platform
# 构建并启动
docker-compose up -d
# 或使用Kubernetes
kubectl apply -f k8s/
```

---

## 📁 项目结构

```
case6-enterprise-platform/
├── app/
│   ├── api/              # API路由
│   ├── core/             # 核心配置
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑
│   └── main.py           # 应用入口
├── tests/                # 测试用例
├── Dockerfile            # Docker镜像
├── docker-compose.yml    # 本地开发
└── k8s/                  # Kubernetes配置
```

---

## 🔑 核心架构

### FastAPI应用结构

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, rag, admin

app = FastAPI(
    title="企业级RAG平台",
    version="1.0.0",
    description="生产级检索增强生成API"
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(auth.router, prefix="/api/v1/auth", tags=["认证"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["管理"])

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    await connect_database()
    await init_redis()
    await load_models()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    await close_connections()
```

### JWT认证

```python
# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/login")
async def login(username: str, password: str):
    """用户登录"""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="认证失败")

    # 生成JWT token
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/refresh")
async def refresh_token(token: str = Depends(oauth2_scheme)):
    """刷新token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)

    # 生成新token
    new_token = create_access_token(data={"sub": username})
    return {"access_token": new_token}
```

### RAG服务

```python
# app/api/rag.py
from fastapi import APIRouter, Depends, BackgroundTasks
from app.services.cache import cache_manager
from app.services.rag import RAGEngine

router = APIRouter()
rag_engine = RAGEngine()

@router.post("/query")
async def query(
    question: str,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """RAG查询接口"""

    # 检查缓存
    cached_result = await cache_manager.get(question)
    if cached_result:
        return {"answer": cached_result, "from_cache": True}

    # 执行RAG查询
    result = await rag_engine.query(question)

    # 异步更新缓存
    background_tasks.add_task(cache_manager.set, question, result)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "from_cache": False
    }

@router.post("/upload-document")
async def upload_document(
    file: UploadFile,
    current_user = Depends(get_current_user)
):
    """上传文档到知识库"""

    # 验证权限
    if not current_user.can_upload:
        raise HTTPException(status_code=403)

    # 保存文件
    file_path = save_upload_file(file)

    # 异步处理文档
    background_tasks.add_task(
        process_document,
        file_path=file_path,
        user_id=current_user.id
    )

    return {"status": "processing", "file_id": file_id}
```

### Redis缓存

```python
# app/services/cache.py
import redis.asyncio as redis
from typing import Optional

class CacheManager:
    def __init__(self):
        self.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            encoding="utf-8",
            decode_responses=True
        )

    async def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600):
        """设置缓存"""
        await self.redis.setex(key, ttl, value)

    async def delete(self, key: str):
        """删除缓存"""
        await self.redis.delete(key)

    async def invalidate_pattern(self, pattern: str):
        """批量删除"""
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
```

---

## 📊 监控配置

### Prometheus指标

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# 暴露指标
Instrumentator().instrument(app).expose(app)
```

### Grafana仪表盘

- 请求QPS
- 响应时间
- 错误率
- 缓存命中率
- Token使用量

---

## 🐳 部署配置

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-platform
  template:
    metadata:
      labels:
        app: rag-platform
    spec:
      containers:
      - name: api
        image: rag-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## 🎓 学习要点

1. **API设计**
   - RESTful最佳实践
   - 异步编程
   - 错误处理

2. **安全认证**
   - JWT机制
   - 权限控制
   - API限流

3. **性能优化**
   - 缓存策略
   - 数据库优化
   - 并发处理

4. **运维部署**
   - 容器化
   - 编排部署
   - 监控告警

---

## 📈 扩展方向

- [ ] WebSocket实时通信
- [ ] Elasticsearch全文检索
- [ ] 分布式追踪（Jaeger）
- [ ] 灰度发布
- [ ] A/B测试框架

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case6-enterprise-platform)**

**[← 返回案例列表](index.md)**
