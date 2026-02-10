# 第17章：Docker容器化

> 将RAG应用容器化，实现一致性和可移植性

---

## 📚 章节概述

本章将学习如何使用Docker将RAG应用容器化，确保应用在任何环境中都能一致运行。

### 学习目标

完成本章后，你将能够：
- ✅ 理解Docker的核心概念
- ✅ 编写高质量的Dockerfile
- ✅ 使用Docker Compose编排多服务应用
- ✅ 优化Docker镜像大小和构建速度
- ✅ 管理容器网络和数据卷

### 预计时间

- 理论学习：60分钟
- 实践操作：90-120分钟
- 总计：约3小时

---

## 1. Docker基础

### 1.1 为什么需要Docker？

**传统部署的问题**：
```
开发环境：Python 3.9 + PostgreSQL 13
测试环境：Python 3.8 + PostgreSQL 12  → 环境不一致！
生产环境：Python 3.10 + PostgreSQL 14
```

**Docker的解决方案**：
```
开发环境：Docker镜像 → 容器运行
测试环境：相同Docker镜像 → 相同运行结果 ✓
生产环境：相同Docker镜像 → 相同运行结果 ✓
```

### 1.2 Docker核心概念

**镜像（Image）**：
- 应用的只读模板
- 包含代码、运行时、库、环境变量
- 分层存储，可复用

**容器（Container）**：
- 镜像的运行实例
- 隔离的运行环境
- 轻量级、快速启动

**Dockerfile**：
- 构建镜像的脚本
- 声明式配置
- 版本控制友好

### 1.3 Docker vs 虚拟机

| 特性 | Docker容器 | 虚拟机 |
|------|-----------|--------|
| 启动速度 | 秒级 | 分钟级 |
| 资源占用 | MB级 | GB级 |
| 性能 | 接近原生 | 有损耗 |
| 隔离性 | 进程级 | 系统级 |
| 可移植性 | 优秀 | 一般 |

---

## 2. 编写Dockerfile

### 2.1 基础Dockerfile

最简单的RAG应用Dockerfile：

```dockerfile
# 使用官方Python镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 最佳实践Dockerfile

优化后的生产级Dockerfile：

```dockerfile
# 多阶段构建
# 阶段1：构建
FROM python:3.10-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖到临时目录
RUN pip install --user --no-cache-dir -r requirements.txt

# 阶段2：运行
FROM python:3.10-slim

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /root/.local /root/.local

# 确保Python能找到安装的包
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 Dockerfile最佳实践

**1. 选择合适的基础镜像**：
```dockerfile
# 好的选择（小而安全）
FROM python:3.10-slim        # 100MB
FROM python:3.10-alpine      # 50MB（更小但可能有兼容性问题）

# 不推荐（太大）
FROM python:3.10             # 900MB
```

**2. 优化层缓存**：
```dockerfile
# ❌ 不好：每次代码变动都要重新安装依赖
COPY . .
RUN pip install -r requirements.txt

# ✅ 好：依赖只在变化时才重新安装
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

**3. 合并RUN指令**：
```dockerfile
# ❌ 不好：创建多层
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

# ✅ 好：合并为单层
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
```

**4. 使用.dockerignore**：
```
# .dockerignore文件
__pycache__
*.pyc
.env
.git
.venv
tests/
*.md
```

**5. 最小化镜像层数**：
```dockerfile
# 使用多阶段构建减少最终镜像大小
# 只保留运行时必需的文件
```

---

## 3. 多服务编排

### 3.1 RAG系统架构

一个典型的RAG系统包含多个服务：

```
┌─────────────────┐
│   Nginx Proxy   │  (反向代理)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼────┐ ┌─▼────────┐
│  RAG   │ │ Vector DB│  (ChromaDB)
│  API   │ └───────────┘
└───┬────┘
    │
┌───▼────────┐
│ PostgreSQL │  (元数据存储)
└────────────┘
```

### 3.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # RAG API服务
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8001
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - chromadb
    volumes:
      - ./app:/app
    networks:
      - rag-network
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    container_name: rag-postgres
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=ragdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rag-network
    restart: unless-stopped

  # ChromaDB向量数据库
  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    networks:
      - rag-network
    restart: unless-stopped

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: rag-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - rag-api
    networks:
      - rag-network
    restart: unless-stopped

# 数据卷
volumes:
  postgres_data:
  chroma_data:

# 网络
networks:
  rag-network:
    driver: bridge
```

### 3.3 Nginx配置

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream rag_api {
        server rag-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 10M;

        location / {
            proxy_pass http://rag_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # 超时设置
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /health {
            proxy_pass http://rag_api/health;
            access_log off;
        }
    }
}
```

### 3.4 环境变量管理

```bash
# .env文件
OPENAI_API_KEY=sk-xxxxx
DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
CHROMA_HOST=chromadb
CHROMA_PORT=8001
LOG_LEVEL=info
```

---

## 4. 镜像优化

### 4.1 镜像大小优化

**问题分析**：
```bash
# 查看镜像层
docker history rag-api:latest

# 查看镜像大小
docker images rag-api
```

**优化技巧**：

1. **使用alpine镜像**：
```dockerfile
# 从900MB减少到100MB
FROM python:3.10-slim  # 或 alpine
```

2. **多阶段构建**：
```dockerfile
# 只保留运行时必需的文件
FROM python:3.10-slim as builder
# ... 构建步骤

FROM python:3.10-slim
COPY --from=builder /app /app
# 最终镜像只包含运行时文件
```

3. **清理缓存**：
```dockerfile
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

4. **使用.dockerignore**：
```
# 排除不必要的文件
.git
.venv
__pycache__
*.pyc
tests/
docs/
```

### 4.2 构建速度优化

```dockerfile
# 1. 利用缓存
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# 2. 并行安装
RUN pip install --no-cache-dir \
    package1 \
    package2 \
    package3

# 3. 使用BuildKit
# DOCKER_BUILDKIT=1 docker build .
```

### 4.3 安全扫描

```bash
# 使用Trivy扫描漏洞
trivy image rag-api:latest

# 修复漏洞
# 更新基础镜像
FROM python:3.10-slim  # 使用最新版本
```

---

## 5. 管理和运维

### 5.1 常用命令

```bash
# 构建镜像
docker build -t rag-api:latest .

# 运行容器
docker run -d -p 8000:8000 --name rag-api rag-api:latest

# 查看日志
docker logs -f rag-api

# 进入容器
docker exec -it rag-api /bin/bash

# 查看资源使用
docker stats rag-api

# 停止容器
docker stop rag-api

# 删除容器
docker rm rag-api

# Compose命令
docker-compose up -d           # 启动服务
docker-compose down           # 停止服务
docker-compose logs -f        # 查看日志
docker-compose ps             # 查看状态
docker-compose restart        # 重启服务
```

### 5.2 数据持久化

```yaml
# docker-compose.yml
services:
  postgres:
    volumes:
      # 命名卷（推荐）
      - postgres_data:/var/lib/postgresql/data

      # 绑定挂载（开发）
      - ./data:/var/lib/postgresql/data

volumes:
  postgres_data:
    driver: local
```

### 5.3 健康检查

```dockerfile
# Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

```python
# FastAPI健康检查端点
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## 6. 实战练习

### 练习1：容器化基础RAG应用

**任务**：
1. 为简单的RAG应用编写Dockerfile
2. 构建镜像并测试
3. 优化镜像大小

**要求**：
- 使用python:3.10-slim基础镜像
- 实现多阶段构建
- 镜像大小 < 200MB

**提示**：
```dockerfile
# Dockerfile模板
FROM python:3.10-slim as builder
# ... 安装依赖

FROM python:3.10-slim
# ... 复制文件和配置
```

---

### 练习2：多服务编排

**任务**：
1. 创建docker-compose.yml
2. 配置RAG API + PostgreSQL + ChromaDB
3. 配置服务间通信
4. 测试完整流程

**要求**：
- 服务正常启动
- 数据持久化
- 容器重启后数据不丢失

**验证**：
```bash
# 启动服务
docker-compose up -d

# 测试API
curl http://localhost/health

# 查看日志
docker-compose logs -f
```

---

### 练习3：生产级配置

**任务**：
1. 配置Nginx反向代理
2. 实现健康检查
3. 配置日志收集
4. 设置资源限制

**要求**：
- 服务高可用
- 自动重启
- 资源限制合理

```yaml
services:
  rag-api:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

---

## 7. 故障排查

### 7.1 常见问题

**问题1：容器无法启动**
```bash
# 查看日志
docker logs rag-api

# 检查配置
docker inspect rag-api

# 进入容器调试
docker run -it --rm rag-api:latest /bin/bash
```

**问题2：网络连接失败**
```bash
# 检查网络
docker network ls
docker network inspect rag-network

# 测试连接
docker exec rag-api ping postgres
```

**问题3：数据丢失**
```bash
# 检查卷
docker volume ls
docker volume inspect postgres_data

# 备份数据
docker exec postgres pg_dump -U user ragdb > backup.sql
```

### 7.2 调试技巧

1. **使用--entrypoint覆盖**：
```bash
docker run --rm -it --entrypoint /bin/bash rag-api:latest
```

2. **查看详细日志**：
```bash
docker logs --tail 100 -f rag-api
```

3. **导出容器文件系统**：
```bash
docker export rag-api > rag-api.tar
```

---

## 8. 总结

### 关键要点

1. **Dockerfile最佳实践**
   - 选择合适的基础镜像
   - 优化层缓存
   - 使用多阶段构建

2. **Docker Compose**
   - 简化多服务管理
   - 统一环境配置
   - 方便本地开发

3. **镜像优化**
   - 减小镜像大小
   - 提升构建速度
   - 增强安全性

4. **生产部署**
   - 健康检查
   - 资源限制
   - 数据持久化

### 下一步

- 学习Kubernetes部署（第18章）
- 了解监控和日志（第19章）
- 实施CI/CD流程（第20章）

---

## 9. 参考资源

### 官方文档

- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

### 推荐工具

- **Trivy**：安全扫描
- **Hadolint**：Dockerfile检查
- **Dive**：镜像分析

### 示例项目

- [Docker Samples](https://github.com/docker/awesome-compose)
- [FastAPI Docker Example](https://fastapi.tiangolo.com/deployment/docker/)

---

**恭喜完成第17章！** 🎉

你已经掌握了Docker容器化的核心技能，可以将RAG应用打包部署了！

**下一步**：第18章 - Kubernetes部署
