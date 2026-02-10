# 模块4 - Jupyter Notebooks

> 生产部署实战的交互式学习环境

---

## 📚 Available Notebooks

### 17_deployment_practice.ipynb
**主题**：Docker和Kubernetes部署实践

**内容**：
- ✅ Docker容器化实践
- ✅ Docker Compose多服务编排
- ✅ Kubernetes部署配置
- ✅ 服务健康检查
- ✅ 滚动更新实践

**学习时间**：90-120分钟

**涵盖技能**：
- 编写生产级Dockerfile
- 配置Docker Compose
- 编写K8s YAML文件
- 部署和监控服务

---

## 🚀 快速开始

### 环境准备

```bash
# 确保已安装
- Python 3.9+
- Docker
- kubectl (可选，用于K8s练习)
```

### 运行Notebook

```bash
# 进入目录
cd 04-生产部署/notebooks

# 启动Jupyter
jupyter notebook

# 或使用JupyterLab
jupyter lab
```

---

## 📝 使用建议

### 学习顺序

1. **先阅读章节文档**：了解理论知识
2. **运行Notebook**：动手实践代码
3. **完成练习题**：巩固知识点
4. **查看参考答案**：验证理解

### 实践建议

- ✅ 逐个运行代码单元格
- ✅ 理解每个命令的作用
- ✅ 尝试修改参数观察效果
- ✅ 遇到错误时阅读错误信息

---

## ⚠️ 注意事项

### Docker练习

- 需要安装Docker Desktop
- 某些命令可能需要sudo权限
- 注意镜像大小和构建时间

### Kubernetes练习

- 需要minikube或kind（本地K8s）
- 或者使用云服务（如EKS、GKE）
- 注意资源配额和成本

### 监控练习

- Prometheus + Grafana需要较多资源
- 建议至少4GB内存
- 可以使用Docker Compose快速启动

---

## 🔧 故障排查

### Docker问题

```bash
# 检查Docker是否运行
docker ps

# 查看Docker日志
docker logs <container>

# 清理未使用的资源
docker system prune -a
```

### K8s问题

```bash
# 查看Pod状态
kubectl get pods -A

# 查看Pod详情
kubectl describe pod <pod-name>

# 查看日志
kubectl logs <pod-name>
```

---

## 📖 相关资源

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Docs](https://prometheus.io/docs/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

**祝你学习愉快！** 🚀
