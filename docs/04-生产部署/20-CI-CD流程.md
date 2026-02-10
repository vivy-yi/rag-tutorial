# 第20章：CI/CD流程

> 自动化测试、构建和部署，实现快速可靠的交付

---

## 📚 章节概述

本章将学习如何构建完整的CI/CD流水线，实现自动化测试、构建和部署。

### 学习目标

完成本章后，你将能够：
- ✅ 理解CI/CD的核心概念
- ✅ 配置GitHub Actions工作流
- ✅ 实现自动化测试
- ✅ 配置自动构建和部署
- ✅ 实施环境管理策略
- ✅ 建立发布流程

### 预计时间

- 理论学习：60分钟
- 实践操作：90-120分钟
- 总计：约3-4小时

---

## 1. CI/CD基础

### 1.1 核心概念

**CI（Continuous Integration）持续集成**：
```
开发者提交代码
    ↓
自动触发构建
    ↓
运行测试套件
    ↓
代码质量检查
    ↓
反馈结果
```

**CD（Continuous Deployment）持续部署**：
```
CI通过
    ↓
构建镜像
    ↓
推送到镜像仓库
    ↓
部署到测试环境
    ↓
自动/手动部署到生产
```

### 1.2 CI/CD价值

- **快速反馈**：及早发现问题
- **降低风险**：小步快跑，频繁发布
- **提高质量**：自动化测试保障
- **节省时间**：减少手动操作

### 1.3 工具选择

**主流CI/CD平台**：
- **GitHub Actions**：与GitHub深度集成
- **GitLab CI/CD**：GitLab内置
- **Jenkins**：灵活强大
- **CircleCI**：简单易用

**本章选择GitHub Actions**：与教程项目最匹配

---

## 2. GitHub Actions基础

### 2.1 工作流结构

```yaml
name: CI/CD Pipeline  # 工作流名称

on:  # 触发条件
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:  # 任务
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=app tests/
```

### 2.2 关键概念

**Workflow（工作流）**：
- 完整的自动化流程
- 由一个或多个Job组成

**Job（任务）**：
- 工作流中的执行单元
- 运行在指定的Runner上

**Step（步骤）**：
- Job中的执行动作
- 可以运行命令或使用Action

**Action（动作）**：
- 可重用的步骤
- GitHub Marketplace提供大量现成的

**Runner（运行器）**：
- 执行Job的服务器
- GitHub托管或自托管

---

## 3. 完整CI配置

### 3.1 代码质量检查

```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.10'

jobs:
  # 代码风格检查
  lint:
    name: Code Style Check
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install linting tools
      run: |
        pip install black flake8 isort mypy

    - name: Run Black check
      run: black --check .

    - name: Run Flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Run isort check
      run: isort --check-only .

    - name: Run mypy
      run: mypy app/

  # 安全扫描
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # 单元测试
  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: [lint, security]
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Run tests with coverage
      env:
        DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
      run: |
        pytest --cov=app --cov-report=xml --cov-report=html --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Archive coverage reports
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: htmlcov/
```

### 3.2 集成测试

```yaml
  # 集成测试
  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest docker-compose

    - name: Start services with Docker Compose
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to be ready

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Cleanup
      if: always()
      run: docker-compose -f docker-compose.test.yml down -v
```

---

## 4. Docker镜像构建

### 4.1 构建和推送

```yaml
  # 构建Docker镜像
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, integration-test]
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: your-dockerhub-username/rag-api
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILD_DATE=${{ github.event.head_commit.timestamp }}
          VCS_REF=${{ github.sha }}

    - name: Image digest
      run: echo ${{ steps.build.outputs.digest }}
```

### 4.2 多架构构建

```yaml
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v2

    - name: Build and push (multi-arch)
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
```

---

## 5. 部署配置

### 5.1 部署到K8s

```yaml
  # 部署到测试环境
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: staging
      url: https://staging.rag.example.com
    if: github.ref == 'refs/heads/develop'

    steps:
    - uses: actions/checkout@v3

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubeconfig
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > $HOME/.kube/config

    - name: Update deployment image
      run: |
        kubectl set image deployment/rag-api \
          rag-api=${{ needs.build.outputs.image-tag }} \
          -n rag-staging

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/rag-api -n rag-staging --timeout=5m

    - name: Verify deployment
      run: |
        kubectl get pods -n rag-staging -l app=rag-api

    - name: Run smoke tests
      run: |
        kubectl run smoke-test --image=curlimages/curl --rm -i --restart=Never \
          -- curl -f http://rag-api-service.rag-staging/health
```

### 5.2 部署到生产环境

```yaml
  # 部署到生产（需手动批准）
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: production
      url: https://rag.example.com
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3

    - name: Configure kubeconfig
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > $HOME/.kube/config

    - name: Create new deployment tag
      run: |
        kubectl patch deployment rag-api -n rag-system -p '{"spec":{"template":{"metadata":{"annotations":{"deployedAt":"'$(date +%s)'"}}}}}'

    - name: Update deployment image
      run: |
        kubectl set image deployment/rag-api \
          rag-api=${{ needs.build.outputs.image-tag }} \
          -n rag-system

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/rag-api -n rag-system --timeout=10m

    - name: Verify deployment
      run: |
        kubectl get pods -n rag-system -l app=rag-api

    - name: Run smoke tests
      run: |
        for i in {1..10}; do
          curl -f https://rag.example.com/health && break || sleep 10
        done

    - name: Notify on success
      if: success()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Production deployment successful!'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}

    - name: Rollback on failure
      if: failure()
      run: |
        kubectl rollout undo deployment/rag-api -n rag-system
```

---

## 6. 环境管理

### 6.1 环境配置

```yaml
# .github/environments/production.yml
name: production
deployment_branches:
  matching_branches:
    - main
protection_rules:
  - required_reviewers:
      - name: senior-dev-1
      - name: senior-dev-2
    required_deployment_approvals: 2
```

### 6.2 Secrets管理

**GitHub Secrets设置**：
```bash
# 在GitHub仓库设置中添加以下Secrets
DOCKER_USERNAME
DOCKER_PASSWORD
KUBE_CONFIG_STAGING
KUBE_CONFIG_PROD
OPENAI_API_KEY
SLACK_WEBHOOK
CODECOV_TOKEN
```

**使用Secrets**：
```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

---

## 7. 发布流程

### 7.1 语义化版本

```yaml
  # 自动创建Release
  release:
    name: Create Release
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/checkout@v3

    - name: Generate changelog
      id: changelog
      uses: metcalfc/changelog-generator@v4.1.0
      with:
        myToken: ${{ secrets.GITHUB_TOKEN }}

    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: ${{ steps.changelog.outputs.changelog }}
        draft: false
        prerelease: false
```

### 7.2 自动版本号

```yaml
  # Bump version
  bump-version:
    name: Bump Version
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: '0'

    - name: Bump version and push tag
      uses: anothrNick/github-tag-action@1.39.0
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        WITH_V: true
        DEFAULT_BUMP: patch
```

---

## 8. 实战练习

### 练习1：配置基础CI

**任务**：
1. 创建`.github/workflows/ci.yml`
2. 配置代码检查
3. 添加单元测试
4. 配置覆盖率报告

**验证**：
```bash
# 推送代码触发CI
git push origin feature/test-ci

# 查看Actions运行
# 访问GitHub仓库的Actions标签
```

---

### 练习2：配置自动构建

**任务**：
1. 配置Docker构建
2. 推送到镜像仓库
3. 实现多架构构建
4. 配置镜像缓存

**验证**：
```bash
# 查看构建的镜像
docker pull your-registry/rag-api:latest
docker images
```

---

### 练习3：配置CD流水线

**任务**：
1. 配置测试环境部署
2. 配置生产环境部署
3. 设置环境保护规则
4. 配置自动回滚

**验证**：
```bash
# 合并到main分支
# 检查自动部署
kubectl get pods -n rag-system
```

---

## 9. 最佳实践

### 9.1 CI/CD最佳实践

- **快速反馈**：优化CI速度
- **并行执行**：独立Job并行运行
- **缓存依赖**：加速构建过程
- **逐步部署**：先测试后生产

### 9.2 安全实践

- **Secrets管理**：使用GitHub Secrets
- **最小权限**：限制token权限
- **审计日志**：记录所有操作
- **定期轮换**：更新密钥

### 9.3 性能优化

- **使用缓存**：依赖缓存、构建缓存
- **并行Job**：独立任务并行
- **条件执行**：只在需要时运行
- **增量构建**：只构建变化的部分

---

## 10. 总结

### 关键要点

1. **CI/CD流程**
   - 自动化测试
   - 自动构建
   - 自动部署

2. **GitHub Actions**
   - 工作流配置
   - 多环境部署
   - Secrets管理

3. **发布管理**
   - 版本策略
   - 发布流程
   - 回滚机制

### 下一步

- 学习性能优化（第21章）
- 安全实践（第22章）

---

**恭喜完成第20章！** 🎉

你已经掌握构建完整CI/CD流水线的技能！

**下一步**：第21章 - 性能优化
