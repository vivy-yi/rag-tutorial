# 贡献指南

感谢你对RAG完整教程的关注！我们欢迎各种形式的贡献。

---

## 🤝 如何贡献

### 报告问题

如果你发现了bug、错误或有疑问：

1. 检查 [Issues](https://github.com/vivy-yi/rag-tutorial/issues) 确保问题还没有被报告
2. 创建一个Issue，详细描述问题
3. 包含复现步骤（如果可能）
4. 提供截图或错误日志

### 提交代码

#### 准备工作

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 确保你的代码符合我们的代码规范

#### 开发流程

1. 编写代码或文档
2. 添加测试（如果适用）
3. 确保所有测试通过
4. 提交代码 (`git commit -m 'Add some AmazingFeature'`)
5. 推送到分支 (`git push origin feature/AmazingFeature`)
6. 开启 Pull Request

#### Pull Request指南

- PR标题应该清晰描述更改内容
- 在PR描述中详细说明做了什么修改以及为什么
- 如果修复了某个Issue，请在描述中关联它（如 `Fixes #123`）
- 保持PR尽可能小，便于review

---

## 📝 代码规范

### Python代码

- 遵循 PEP 8 规范
- 使用有意义的变量名和函数名
- 添加必要的注释和文档字符串
- 最大行长度：100字符

### 文档

- 使用Markdown格式
- 添加适当的标题层级
- 包含代码示例
- 添加必要的图表和说明

### Jupyter Notebooks

- 确保notebook可以从头到尾运行
- 添加必要的说明文字
- 清理不必要的输出
- 包含预期的输出示例

---

## 🎯 贡献方向

我们特别欢迎以下方面的贡献：

### 内容改进

- 修正错误或不准确的内容
- 添加更多示例和练习题
- 完善实战案例
- 添加新的章节或模块

### 代码改进

- 优化现有代码
- 添加测试用例
- 改进文档注释
- 性能优化

### Bug修复

- 修复发现的bug
- 解决兼容性问题
- 优化错误处理

### 新功能

- 添加新的RAG技术实现
- 集成新的工具或框架
- 添加新的评估方法

---

## 📋 开发环境设置

```bash
# 1. Fork并克隆仓库
git clone https://github.com/vivy-yi/rag-tutorial.git
cd rag-tutorial

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# 4. 安装pre-commit hooks（可选）
pip install pre-commit
pre-commit install
```

---

## ✅ 提交前检查清单

在提交PR前，请确保：

- [ ] 代码符合项目的代码规范
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] README.md中的必要信息已更新

---

## 📧 联系方式

如果你有任何问题或建议：

- **GitHub Issues**: [提交Issue](https://github.com/vivy-yi/rag-tutorial/issues)
- **Email**: your.email@example.com

---

## 🌟 成为贡献者

所有贡献者将被添加到项目的贡献者列表中。

感谢你的贡献！🙏
