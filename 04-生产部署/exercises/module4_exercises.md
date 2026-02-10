# 模块4练习题 - 生产部署实战

> 巩固生产部署技能的实战练习

---

## 📋 练习概览

本模块包含**15道练习题**，覆盖以下主题：
- Docker容器化（3题）
- Kubernetes部署（3题）
- 监控和日志（3题）
- CI/CD流程（3题）
- 性能和安全（3题）

---

## 第一部分：Docker容器化（3题）

### 练习1：优化Dockerfile

**任务**：
给定以下Dockerfile，存在多个问题，请优化它：

```dockerfile
# 问题Dockerfile
FROM python:latest

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
```

**要求**：
1. 选择更合适的基础镜像
2. 优化层缓存
3. 添加非root用户
4. 减小镜像大小

**提交**：
- 优化后的Dockerfile
- 镜像大小对比
- 构建时间对比

---

### 练习2：多服务Docker Compose

**任务**：
为以下RAG系统创建完整的docker-compose.yml：

**服务**：
1. RAG API（FastAPI）
2. PostgreSQL数据库
3. ChromaDB向量数据库
4. Nginx反向代理

**要求**：
- 配置服务间通信
- 设置数据卷持久化
- 配置环境变量
- 设置健康检查
- 添加网络配置

**提交**：
- docker-compose.yml文件
- docker-compose.prod.yml（生产配置）
- 启动命令说明

---

### 练习3：Docker安全扫描

**任务**：
使用Trivy扫描以下镜像并修复漏洞：

1. `python:3.10-slim`
2. `postgres:15`
3. 你自己构建的镜像

**要求**：
1. 安装Trivy
2. 扫描镜像并生成报告
3. 修复高危漏洞
4. 重新扫描验证

**提交**：
- 扫描报告
- 修复措施说明
- 修复后扫描结果

---

## 第二部分：Kubernetes部署（3题）

### 练习4：部署RAG API到K8s

**任务**：
为RAG API创建完整的K8s部署配置：

**要求**：
1. Deployment配置（包含健康检查）
2. Service配置
3. ConfigMap和Secret
4. Ingress配置
5. HPA配置

**提交**：
- 所有YAML文件
- 部署命令
- 验证步骤

---

### 练习5：StatefulSet部署PostgreSQL

**任务**：
使用StatefulSet部署PostgreSQL数据库：

**要求**：
1. StatefulSet配置
2. Headless Service
3. PersistentVolumeClaim
4. 数据备份策略

**提交**：
- 完整配置文件
- 部署和验证步骤
- 备份脚本

---

### 练习6：零停机部署

**任务**：
配置RAG API的滚动更新策略：

**要求**：
1. 配置滚动更新参数
2. 设置maxSurge和maxUnavailable
3. 配置readinessProbe
4. 测试滚动更新
5. 实现快速回滚

**提交**：
- Deployment配置
- 更新策略说明
- 测试结果

---

## 第三部分：监控和日志（3题）

### 练习7：配置Prometheus监控

**任务**：
为RAG系统配置完整的Prometheus监控：

**要求**：
1. 在应用中添加Prometheus指标
2. 配置Prometheus采集
3. 创建Grafana仪表盘
4. 配置关键告警规则

**指标**：
- QPS
- P95延迟
- 错误率
- CPU/内存使用

**提交**：
- 指标采集代码
- Prometheus配置
- Grafana仪表盘JSON
- 告警规则

---

### 练习8：ELK日志系统

**任务**：
搭建ELK日志聚合系统：

**要求**：
1. 部署Elasticsearch
2. 配置Filebeat采集日志
3. 配置Logstash处理
4. 创建Kibana索引模式
5. 配置日志仪表盘

**提交**：
- Docker Compose配置
- Filebeat配置
- Kibana仪表盘截图

---

### 练习9：分布式追踪

**任务**：
为RAG系统实现分布式追踪：

**要求**：
1. 集成OpenTelemetry
2. 部署Jaeger
3. 追踪完整请求链路
4. 分析性能瓶颈

**提交**：
- OpenTelemetry集成代码
- Jaeger配置
- 追踪结果分析

---

## 第四部分：CI/CD流程（3题）

### 练习10：GitHub Actions工作流

**任务**：
创建完整的CI/CD流水线：

**阶段**：
1. Lint和格式检查
2. 单元测试
3. Docker镜像构建
4. 安全扫描
5. 部署到测试环境

**提交**：
- .github/workflows/ci.yml
- 配置说明
- 运行结果

---

### 练习11：自动化测试

**任务**：
为RAG系统添加自动化测试：

**要求**：
1. 单元测试（pytest）
2. 集成测试
3. E2E测试
4. 性能测试
5. 测试覆盖率报告

**提交**：
- 测试代码
- 配置文件
- 覆盖率报告

---

### 练习12：多环境部署

**任务**：
配置多环境自动部署：

**环境**：
1. Development（自动部署）
2. Staging（PR合并后部署）
3. Production（手动批准）

**要求**：
- 环境配置隔离
- Secrets管理
- 部署策略

**提交**：
- GitHub Actions配置
- 环境配置说明

---

## 第五部分：性能和安全（3题）

### 练习13：实施多层缓存

**任务**：
为RAG系统实现多层缓存：

**层级**：
1. 应用内存缓存
2. Redis缓存
3. CDN缓存（可选）

**要求**：
- 缓存查询结果
- 缓存嵌入向量
- 实现缓存失效
- 监控命中率

**提交**：
- 缓存实现代码
- 性能测试结果
- 命中率监控

---

### 练习14：API安全加固

**任务**：
实施以下安全措施：

**要求**：
1. JWT认证
2. 速率限制
3. 输入验证
4. SQL注入防护
5. XSS防护

**提交**：
- 安全实现代码
- 测试用例
- 安全扫描报告

---

### 练习15：性能优化和压力测试

**任务**：
优化RAG系统性能并进行压力测试：

**优化项**：
1. 数据库查询优化
2. 向量检索优化
3. 批处理实现
4. 连接池配置

**压力测试**：
1. 使用Locust进行测试
2. 测试不同并发级别
3. 识别性能瓶颈
4. 优化前后对比

**提交**：
- 优化代码
- 压力测试脚本
- 性能对比报告

---

## 📊 练习难度

| 练习 | 难度 | 预计时间 |
|------|------|----------|
| 1-3（Docker） | ⭐⭐ | 2-3小时 |
| 4-6（K8s） | ⭐⭐⭐ | 3-4小时 |
| 7-9（监控） | ⭐⭐⭐ | 3-4小时 |
| 10-12（CI/CD） | ⭐⭐⭐⭐ | 4-5小时 |
| 13-15（性能安全） | ⭐⭐⭐⭐ | 4-5小时 |

**总计**：约16-21小时

---

## 🎯 学习建议

### 循序渐进

1. **先易后难**：从Docker练习开始
2. **理解原理**：不要只照搬代码
3. **实际操作**：每个练习都要动手
4. **记录问题**：建立自己的故障排查手册

### 实践建议

- 使用本地环境先测试
- 每完成一个练习就提交代码
- 记录遇到的问题和解决方案
- 与参考答案对比学习

---

## ✅ 提交检查清单

完成练习后，确保：

- [ ] 代码可以正常运行
- [ ] 符合题目所有要求
- [ ] 有适当的注释
- [ ] 包含必要的文档
- [ ] 通过安全扫描
- [ ] 性能达到预期

---

## 📚 参考资源

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

**祝你练习顺利！** 💪
