# RAG完整教程 - 从入门到生产部署

> 🚀 **完整的RAG技术教程** - 从基础概念到生产部署，系统化掌握检索增强生成技术

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/vivy-yi/rag-tutorial?style=social)](https://github.com/vivy-yi/rag-tutorial/stargazers)

**在线文档** | **Jupyter Notebooks** | **实战案例**
---|---|---
[GitHub Pages](https://vivy-yi.github.io/rag-tutorial/) | [查看Notebooks](#-jupyter-notebooks) | [6个完整案例](#-实战案例)

---

## ✨ 特性

- 📚 **系统化学习路径**: 4个模块，20章内容，从入门到精通
- 💻 **17个Jupyter Notebooks**: 交互式学习环境
- 🎯 **6个完整实战案例**: 企业级代码实现
- 📊 **89张技术图表**: 深入理解架构和原理
- ✅ **30+练习题**: 巩固学习成果

---

## 📖 教程大纲

### 模块1：基础入门 (5章)

```bash
01-基础入门/
├── 01-RAG技术概述.md           # RAG技术发展历程
├── 02-环境搭建与工具准备.md      # 开发环境配置
├── 03-基础RAG实现.md            # 第一个RAG系统
├── 04-RAG评估基础.md            # 评估指标和方法
└── 05-模块1总结与项目.md        # 综合项目
```

### 模块2：核心优化 (8章)

```bash
02-核心优化/
├── 06-嵌入模型深入.md            # Transformer嵌入
├── 07-高级分块策略.md            # 智能文档分块
├── 08-查询增强技术.md            # HyDE等技术
├── 09-混合检索与重排序.md        # Vector + BM25
├── 10-高级RAG模式.md            # 迭代、自适应检索
├── 11-性能优化.md                # 系统性能优化
├── 13-检索压缩优化.md ⭐         # 上下文压缩
└── 12-综合项目优化.md            # Intel优化案例
```

### 模块3：高级架构 (4章)

```bash
03-高级架构/
├── 13-Agentic-RAG基础.md        # ReAct Agent
├── 14-高级Agent模式.md ⭐        # Deep Research + 多Agent
├── 15-知识图谱RAG.md            # GraphRAG实现
└── 16-多模态RAG.md              # 图文检索
```

### 模块4：生产部署 (5章)

```bash
04-生产部署/
├── 17-环境配置.md                # 生产环境
├── 18-Docker部署.md             # 容器化部署
├── 19-监控和日志.md              # 可观测性
├── 20-安全实践.md                # 安全最佳实践
└── 22-最佳实践.md                # 生产级建议
```

---

## 💻 Jupyter Notebooks

### 如何使用

```bash
# 1. 克隆仓库
git clone https://github.com/vivy-yi/rag-tutorial.git
cd rag-tutorial

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动Jupyter
jupyter notebook

# 或使用JupyterLab
jupyter lab
```

### Notebook列表

#### 模块1 - 基础入门

| Notebook | 说明 |
|----------|------|
| `01_rag_concepts.ipynb` | RAG核心概念 |
| `02_environment_setup.ipynb` | 环境搭建 |
| `03_basic_rag_implementation.ipynb` | 基础RAG实现 |
| `04_rag_evaluation.ipynb` | RAG评估 |

#### 模块2 - 核心优化

| Notebook | 说明 |
|----------|------|
| `06_embedding_models.ipynb` | 嵌入模型对比 |
| `07_advanced_chunking.ipynb` | 高级分块策略 |
| `08_query_enhancement.ipynb` | 查询增强技术 |
| `09_hybrid_retrieval.ipynb` | 混合检索 |
| `10_advanced_rag_patterns.ipynb` | 高级RAG模式 |
| `11_performance_optimization.ipynb` | 性能优化 |
| `12_comprehensive_optimization.ipynb` | 综合优化 |
| `13_retrieval_compression.ipynb` | 检索压缩 ⭐ |

#### 模块3 - 高级架构

| Notebook | 说明 |
|----------|------|
| `13_react_agent.ipynb` | ReAct Agent |
| `14_advanced_agents.ipynb` | 高级Agent模式 |
| `14_deep_research_agent.ipynb` | Deep Research ⭐ |
| `15_graph_rag.ipynb` | GraphRAG |
| `16_multimodal_rag.ipynb` | 多模态RAG |

---

## 🎯 实战案例

### 案例1：智能客服RAG系统

```bash
实战案例/案例1-智能客服RAG系统/
├── main.py                      # Streamlit界面
├── rag_system.py               # RAG核心实现
├── knowledge_base.py           # 知识库管理
└── requirements.txt            # 依赖包
```

**特点**：
- 基础RAG应用
- 支持多轮对话
- Streamlit Web界面

### 案例2：技术文档问答系统

```bash
实战案例/案例2-技术文档问答系统/
├── main.py
├── hybrid_retriever.py        # 混合检索器
├── reranker.py                # CrossEncoder重排序
└── doc_qa_system.py
```

**特点**：
- Vector + BM25混合检索
- CrossEncoder重排序
- 代码高亮显示

### 案例3：AI研究助手Agent

```bash
实战案例/案例3-AI研究助手Agent/
├── main.py
├── research_agent.py          # ReAct Agent
└── tools.py                    # 工具集成
```

**特点**：
- ReAct Agent模式
- ArXiv论文搜索
- 自动生成报告

### 案例4：企业知识图谱问答

```bash
实战案例/案例4-企业知识图谱问答系统/
├── main.py
├── graph_rag.py               # GraphRAG系统
└── knowledge_graph.py         # 知识图谱构建
```

**特点**：
- GraphRAG实现
- 多跳推理
- 路径可视化

### 案例5：多模态产品问答

```bash
实战案例/案例5-多模态产品问答系统/
├── main.py
└── multimodal_rag.py          # 多模态RAG系统
```

**特点**：
- 图文混合检索
- CLIP/GPT-4V支持

### 案例6：企业级RAG平台

```bash
实战案例/案例6-企业级RAG平台/
├── main.py                     # FastAPI后端
├── rag_engine.py               # RAG引擎
├── auth.py                     # JWT认证
└── cache.py                    # Redis缓存
```

**特点**：
- FastAPI RESTful API
- JWT认证
- Redis缓存
- 用户权限管理

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **框架** | LangChain, LlamaIndex |
| **LLM** | OpenAI GPT-4, Anthropic Claude |
| **向量数据库** | ChromaDB, Pinecone, Milvus |
| **嵌入模型** | OpenAI Embeddings, HuggingFace |
| **Web框架** | Streamlit, FastAPI |
| **部署** | Docker, Kubernetes |

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- pip 或 conda

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/vivy-yi/rag-tutorial.git
cd rag-tutorial

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 学习路径

#### 路径1：快速入门 (2-3周)

```
01-基础入门 → 案例1(智能客服) → 04-生产部署
```

#### 路径2：系统学习 (6-8周)

```
01-基础入门 → 02-核心优化 → 03-高级架构 → 案例1-2-3
```

#### 路径3：专家级 (10-12周)

```
所有模块 → 所有案例 → 深入优化 → 生产部署
```

---

## 📊 教程统计

| 项目 | 数量 |
|------|------|
| **章节数** | 20章 |
| **字数** | ~500,000 |
| **Jupyter Notebooks** | 17个 |
| **实战案例** | 6个 |
| **Python文件** | 25+ |
| **练习题** | 30+道 |
| **图片资源** | 89张 |

---

## 📝 内容结构

```
RAG完整教程/
├── 01-基础入门/              # 模块1：基础入门
│   ├── *.md                  # 章节文档
│   ├── notebooks/            # Jupyter Notebooks
│   └── exercises/            # 练习题和答案
│
├── 02-核心优化/              # 模块2：核心优化
│   ├── *.md
│   ├── notebooks/
│   └── exercises/
│
├── 03-高级架构/              # 模块3：高级架构
│   ├── *.md
│   ├── notebooks/
│   └── exercises/
│
├── 04-生产部署/              # 模块4：生产部署
│   ├── *.md
│   ├── notebooks/
│   └── exercises/
│
├── 实战案例/                 # 6个完整案例
│   ├── 案例1-智能客服RAG系统/
│   ├── 案例2-技术文档问答系统/
│   ├── 案例3-AI研究助手Agent/
│   ├── 案例4-企业知识图谱问答系统/
│   ├── 案例5-多模态产品问答系统/
│   └── 案例6-企业级RAG平台/
│
├── images/                   # 图片资源
│   ├── module1-basic/
│   ├── module2-optimization/
│   ├── module3-advanced/
│   ├── module4-production/
│   └── logos/
│
├── README.md                 # 本文件
├── .gitignore                # Git忽略配置
├── requirements.txt          # Python依赖
└── LICENSE                   # MIT许可证
```

---

## 🎓 学习建议

### 循序渐进

1. **先掌握基础**：完成模块1
2. **学习优化技术**：学习模块2的核心内容
3. **实践高级架构**：探索模块3的Agent和GraphRAG
4. **生产部署**：了解模块4的部署方案

### 理论与实践结合

- 每章都有对应的Jupyter Notebook
- 边学边练，完成练习题
- 运行实战案例，理解生产级实现

### 加入社区

- 提出Issue反馈问题
- 提交PR改进教程
- 分享你的学习心得

---

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本教程整合了多个优质开源项目的资源：

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的LLM应用开发框架
- [AgenticRAG-Survey](https://github.com/AutonLab/AgenticRAG-Survey) - Agentic RAG调研
- [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) - RAG技术合集
- [advanced-rag](https://github.com/langchain-ai/rag-from-scratch) - RAG从零开始

感谢所有贡献者！

---

## 📮 联系方式

- **Issue**: [GitHub Issues](https://github.com/vivy-yi/rag-tutorial/issues)
- **Email**: your.email@example.com

---

## ⭐ 如果这个教程对你有帮助

请给个Star支持一下！🙏

[![GitHub stars](https://img.shields.io/github/stars/vivy-yi/rag-tutorial?style=social)](https://github.com/vivy-yi/rag-tutorial/stargazers)
