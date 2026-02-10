# 第18章：Kubernetes部署

> 在K8s上部署和管理RAG系统，实现高可用和自动扩展

---

## 📚 章节概述

本章将学习如何使用Kubernetes部署和管理RAG系统，实现生产级别的可用性和可扩展性。

### 学习目标

完成本章后，你将能够：
- ✅ 理解Kubernetes核心概念
- ✅ 编写K8s资源配置文件
- ✅ 部署RAG系统到K8s集群
- ✅ 配置自动扩展（HPA）
- ✅ 实现滚动更新和回滚
- ✅ 管理配置和密钥

### 预计时间

- 理论学习：90分钟
- 实践操作：120-150分钟
- 总计：约4-5小时

---

## 1. Kubernetes基础

### 1.1 为什么需要Kubernetes？

**Docker Compose的局限**：
- 单机部署
- 手动扩展
- 缺少自愈能力
- 无负载均衡

**Kubernetes的优势**：
```
✅ 自动扩展（水平/垂直）
✅ 自愈能力（自动重启）
✅ 负载均衡（Service）
✅ 滚动更新（零停机）
✅ 多节点集群
✅ 声明式配置
```

### 1.2 核心概念

**Pod**：
- 最小部署单元
- 一个或多个容器
- 共享网络和存储
- 易失的（可被替换）

**Deployment**：
- 管理Pod副本
- 声明式更新
- 滚动更新和回滚

**Service**：
- Pod的稳定网络标识
- 负载均衡
- 服务发现

**ConfigMap/Secret**：
- 配置管理
- 敏感数据存储

**Ingress**：
- HTTP路由规则
- 外部访问入口

### 1.3 K8s架构

```
┌─────────────────────────────────────┐
│           Control Plane             │
│  ┌───────────┐  ┌──────────────┐   │
│  │ API Server│  │    Scheduler │   │
│  └─────┬─────┘  └──────┬───────┘   │
│        │                │           │
│  ┌─────▼─────┐  ┌──────▼───────┐   │
│  │Controller │  │ Cloud Controller│ │
│  │ Manager   │  │   Manager     │   │
│  └───────────┘  └──────────────┘   │
└─────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼───┐ ┌──▼────┐
│ Node1 │ │ Node2│ │ Node3 │
│       │ │      │ │       │
│Pod 1  │ │Pod 2 │ │Pod 3  │
│Pod 2  │ │Pod 3 │ │Pod 4  │
└───────┘ └──────┘ └───────┘
```

---

## 2. 部署配置

### 2.1 命名空间

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
    environment: production
```

```bash
# 应用配置
kubectl apply -f namespace.yaml

# 设置默认命名空间
kubectl config set-context --current --namespace=rag-system
```

### 2.2 ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  # 应用配置
  APP_ENV: "production"
  LOG_LEVEL: "info"
  MAX_TOKENS: "2000"

  # 数据库配置（非敏感）
  DB_HOST: "postgres-service"
  DB_PORT: "5432"
  DB_NAME: "ragdb"

  # 向量数据库配置
  CHROMA_HOST: "chromadb-service"
  CHROMA_PORT: "8000"

  # API配置
  API_PORT: "8000"
  WORKERS: "4"
```

### 2.3 Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secret
  namespace: rag-system
type: Opaque
data:
  # Base64编码的敏感信息
  DATABASE_PASSWORD: cGFzc3dvcmQxMjM=  # "password123"
  OPENAI_API_KEY: c2stYWJjZGVmZ2hpams=  # "sk-abcdefghijk"

---
# 使用kubectl创建secret
# echo -n "password123" | base64
# kubectl create secret generic rag-secret \
#   --from-literal=DATABASE_PASSWORD=password123 \
#   --from-literal=OPENAI_API_KEY=sk-xxxxx
```

### 2.4 Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 最多多1个Pod
      maxUnavailable: 1  # 最多不可用1个Pod
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
        version: v1
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP

        # 环境变量
        env:
        - name: DATABASE_URL
          value: "postgresql://$(DB_USER):$(DATABASE_PASSWORD)@$(DB_HOST):$(DB_PORT)/$(DB_NAME)"
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: DB_HOST
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secret
              key: DATABASE_PASSWORD
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secret
              key: OPENAI_API_KEY

        # 资源限制
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # 启动探针
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30

      # 镜像拉取密钥（如果使用私有仓库）
      imagePullSecrets:
      - name: regcred
```

### 2.5 Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
  labels:
    app: rag-api
spec:
  type: ClusterIP
  selector:
    app: rag-api
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 8000
  sessionAffinity: ClientIP  # 会话保持（可选）
```

### 2.6 Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
spec:
  tls:
  - hosts:
    - rag.example.com
    secretName: rag-tls
  rules:
  - host: rag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

---

## 3. 数据库部署

### 3.1 PostgreSQL StatefulSet

```yaml
# postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: rag-system
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: "ragdb"
        - name: POSTGRES_USER
          value: "raguser"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secret
              key: DATABASE_PASSWORD
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - raguser
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - raguser
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

### 3.2 PostgreSQL Service

```yaml
# postgres-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: rag-system
spec:
  clusterIP: None  # Headless service for StatefulSet
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

---

## 4. 自动扩展

### 4.1 水平Pod自动扩展（HPA）

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### 4.2 垂直Pod自动扩展（VPA）

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: rag-api-vpa
  namespace: rag-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: rag-api
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
      maxAllowed:
        cpu: "2000m"
        memory: "4Gi"
      controlledResources: ["cpu", "memory"]
```

---

## 5. 部署和管理

### 5.1 完整部署流程

```bash
# 1. 创建命名空间
kubectl apply -f namespace.yaml

# 2. 创建配置和密钥
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# 3. 部署数据库
kubectl apply -f postgres-statefulset.yaml
kubectl apply -f postgres-service.yaml

# 4. 等待数据库就绪
kubectl wait --for=condition=ready pod -l app=postgres -n rag-system --timeout=300s

# 5. 部署应用
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 6. 部署Ingress（可选）
kubectl apply -f ingress.yaml

# 7. 配置自动扩展
kubectl apply -f hpa.yaml

# 8. 验证部署
kubectl get all -n rag-system
```

### 5.2 查看和调试

```bash
# 查看Pod状态
kubectl get pods -n rag-system
kubectl describe pod rag-api-xxxxx -n rag-system

# 查看日志
kubectl logs rag-api-xxxxx -n rag-system
kubectl logs -f rag-api-xxxxx -n rag-system  # 实时跟踪

# 进入容器
kubectl exec -it rag-api-xxxxx -n rag-system -- /bin/bash

# 查看事件
kubectl get events -n rag-system --sort-by='.lastTimestamp'

# 查看资源使用
kubectl top pods -n rag-system
kubectl top nodes

# 端口转发（本地测试）
kubectl port-forward svc/rag-api-service 8000:80 -n rag-system
```

### 5.3 更新和回滚

```bash
# 更新镜像
kubectl set image deployment/rag-api rag-api=rag-api:v2 -n rag-system

# 查看滚动更新状态
kubectl rollout status deployment/rag-api -n rag-system

# 查看更新历史
kubectl rollout history deployment/rag-api -n rag-system

# 回滚到上一个版本
kubectl rollout undo deployment/rag-api -n rag-system

# 回滚到指定版本
kubectl rollout undo deployment/rag-api --to-revision=2 -n rag-system

# 扩缩容
kubectl scale deployment/rag-api --replicas=5 -n rag-system
```

---

## 6. 存储和持久化

### 6.1 StorageClass

```yaml
# storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
```

### 6.2 PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-data-pvc
  namespace: rag-system
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 20Gi
```

### 6.3 在Pod中使用PVC

```yaml
volumeMounts:
- name: data-volume
  mountPath: /app/data

volumes:
- name: data-volume
  persistentVolumeClaim:
    claimName: rag-data-pvc
```

---

## 7. 实战练习

### 练习1：部署完整RAG系统

**任务**：
1. 创建所有K8s资源文件
2. 部署RAG API + PostgreSQL
3. 配置Service和Ingress
4. 测试完整流程

**验证**：
```bash
# 检查所有Pod运行
kubectl get pods -n rag-system

# 测试API
curl http://rag.example.com/health

# 检查扩展
kubectl get hpa -n rag-system
```

---

### 练习2：配置自动扩展

**任务**：
1. 配置HPA
2. 生成负载测试
3. 观察自动扩展
4. 测试自动缩容

**负载测试**：
```bash
# 使用hey进行负载测试
hey -n 1000 -c 50 http://rag.example.com/query

# 观察Pod数量变化
watch kubectl get pods -n rag-system
```

---

### 练习3：实现零停机部署

**任务**：
1. 配置滚动更新策略
2. 部署新版本
3. 验证无停机
4. 测试回滚

**验证**：
```bash
# 监控更新过程
kubectl rollout status deployment/rag-api -n rag-system

# 持续测试可用性
while true; do curl -f http://rag.example.com/health || break; sleep 1; done
```

---

## 8. 故障排查

### 8.1 Pod问题

**Pod处于Pending状态**：
```bash
# 查看原因
kubectl describe pod rag-api-xxxxx -n rag-system

# 常见原因：
# - 资源不足（Node压力）
# - 镜像拉取失败
# - PVC未绑定
# - 调度器限制
```

**Pod处于CrashLoopBackOff状态**：
```bash
# 查看日志
kubectl logs rag-api-xxxxx -n rag-system

# 检查配置
kubectl describe pod rag-api-xxxxx -n rag-system

# 常见原因：
# - 应用启动失败
# - 健康检查失败
# - 配置错误
# - 依赖服务不可用
```

### 8.2 网络问题

```bash
# 测试Pod间连通性
kubectl exec -it rag-api-xxxxx -n rag-system -- ping postgres-service

# 测试DNS解析
kubectl exec -it rag-api-xxxxx -n rag-system -- nslookup postgres-service

# 查看Service端点
kubectl get endpoints -n rag-system
```

### 8.3 性能问题

```bash
# 查看资源使用
kubectl top pods -n rag-system
kubectl top nodes

# 查看事件
kubectl get events -n rag-system --sort-by='.lastTimestamp'

# 分析性能瓶颈
kubectl logs rag-api-xxxxx -n rag-system --previous
```

---

## 9. 最佳实践

### 9.1 资源管理

```yaml
# 合理设置资源限制
resources:
  requests:
    memory: "512Mi"   # 保证基本运行
    cpu: "250m"
  limits:
    memory: "2Gi"     # 防止资源耗尽
    cpu: "1000m"
```

### 9.2 健康检查

```yaml
# 多层次检查
livenessProbe:   # 存活检查（失败则重启）
readinessProbe:  # 就绪检查（失败则不接收流量）
startupProbe:    # 启动检查（慢启动应用）
```

### 9.3 安全性

```yaml
# 使用非root用户
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

# 限制权限
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
```

### 9.4 可观测性

```yaml
# 添加标签
metadata:
  labels:
    app: rag-api
    version: v1
    environment: production

# 添加注解
metadata:
  annotations:
    description: "RAG API Service"
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
```

---

## 10. 总结

### 关键要点

1. **K8s核心概念**
   - Pod、Deployment、Service
   - ConfigMap、Secret
   - Ingress、HPA

2. **部署流程**
   - 声明式配置
   - 滚动更新
   - 健康检查

3. **运维管理**
   - 自动扩展
   - 故障自愈
   - 资源管理

4. **生产实践**
   - 安全配置
   - 监控告警
   - 备份恢复

### 下一步

- 学习监控和日志（第19章）
- 实施CI/CD流程（第20章）
- 性能优化（第21章）

---

## 11. 参考资源

### 官方文档

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kubernetes API Reference](https://kubernetes.io/docs/reference/)
- [kubectl Command Reference](https://kubernetes.io/docs/reference/kubectl/)

### 推荐工具

- **kubectl**：命令行工具
- **kubectx/kubens**：上下文切换
- **k9s**：终端UI
- **Lens**：GUI管理工具

### 学习资源

- [Kubernetes 101](https://kubernetes.io/docs/tutorials/)
- [Production Patterns](https://kubernetes.io/docs/concepts/cluster-administration/)
- [Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)

---

**恭喜完成第18章！** 🎉

你已经掌握了在Kubernetes上部署RAG系统的完整技能！

**下一步**：第19章 - 监控和日志
