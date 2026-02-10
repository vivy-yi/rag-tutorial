# RAG完整教程 - 从入门到生产部署

> 🚀 **最全面的中文RAG技术教程** - 从基础概念到生产部署，系统化掌握检索增强生成（Retrieval-Augmented Generation）技术。涵盖LangChain、LlamaIndex、向量数据库、Agent、GraphRAG等前沿技术。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/vivy-yi/rag-tutorial?style=social)](https://github.com/vivy-yi/rag-tutorial/stargazers)

**在线文档** | **Jupyter Notebooks** | **实战案例**
---|---|---
[GitHub Pages](https://vivy-yi.github.io/rag-tutorial/) | [查看Notebooks](#-jupyter-notebooks) | [6个完整案例](#-实战案例)

---

## 🔍 什么是RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种结合了检索和生成的AI技术，通过从外部知识库中检索相关信息来增强大语言模型（LLM）的生成能力，有效解决幻觉问题、知识过时和事实错误。

**本教程将带您从零开始，逐步掌握RAG技术，最终能够独立构建企业级RAG应用。**

---

## ✨ 特性

- 📚 **系统化学习路径**: 4个模块，20章内容，从入门到精通
- 💻 **17个Jupyter Notebooks**: 交互式学习环境，即学即练
- 🎯 **6个完整实战案例**: 企业级代码实现（智能客服、文档问答、AI研究助手等）
- 📊 **89张技术图表**: 深入理解架构和原理
- ✅ **30+练习题**: 巩固学习成果，附带详细参考答案
- 🚀 **前沿技术覆盖**: HyDE、Self-RAG、CRAG、GraphRAG、Deep Research、检索压缩等
- 🛠️ **完整技术栈**: LangChain、LlamaIndex、OpenAI、ChromaDB、Pinecone、Streamlit、FastAPI等
- 🌏 **中文优化**: 专为中文学习者设计，案例贴合实际应用场景

---

## 📖 教程大纲

### 模块1：基础入门 (5章)

```bash
docs/01-基础入门/
├── 00-教程导航.md
├── 01-RAG技术概述.md           # RAG技术发展历程
├── 02-环境搭建与工具准备.md      # 开发环境配置
├── 03-基础RAG实现.md            # 第一个RAG系统
├── 04-RAG评估基础.md            # 评估指标和方法
└── 05-模块1总结与项目.md        # 综合项目
```

### 模块2：核心优化 (8章)

```bash
docs/02-核心优化/
├── 06-嵌入模型深入.md            # Transformer嵌入
├── 07-高级分块策略.md            # 智能文档分块
├── 08-查询增强技术.md            # HyDE等技术
├── 09-混合检索与重排序.md        # Vector + BM25
├── 10-高级RAG模式.md            # 迭代、自适应检索
├── 11-性能优化.md                # 系统性能优化
├── 12-综合项目优化.md            # Intel优化案例
└── 13-检索压缩优化.md ⭐         # 上下文压缩
```

### 模块3：高级架构 (4章)

```bash
docs/03-高级架构/
├── 13-Agentic-RAG基础.md        # ReAct Agent
├── 14-高级Agent模式.md ⭐        # Deep Research + 多Agent
├── 15-知识图谱RAG.md            # GraphRAG实现
└── 16-多模态RAG.md              # 图文检索
```

### 模块4：生产部署 (5章)

```bash
docs/04-生产部署/
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

# 4. 打开notebooks目录
cd notebooks
```

### Notebook列表

#### 模块1 - 基础入门 (notebooks/module1/)

| Notebook | 说明 |
|----------|------|
| `01_rag_concepts.ipynb` | RAG核心概念 |
| `02_environment_setup.ipynb` | 环境搭建 |
| `03_basic_rag_implementation.ipynb` | 基础RAG实现 |
| `04_rag_evaluation.ipynb` | RAG评估 |

#### 模块2 - 核心优化 (notebooks/module2/)

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

#### 模块3 - 高级架构 (notebooks/module3/)

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

**路径**: `projects/case1-customer-service/`

**特点**：
- 基础RAG应用
- 支持多轮对话
- Streamlit Web界面

### 案例2：技术文档问答系统

**路径**: `projects/case2-doc-qa/`

**特点**：
- Vector + BM25混合检索
- CrossEncoder重排序
- 代码高亮显示

### 案例3：AI研究助手Agent

**路径**: `projects/case3-research-agent/`

**特点**：
- ReAct Agent模式
- ArXiv论文搜索
- 自动生成报告

### 案例4：企业知识图谱问答

**路径**: `projects/case4-knowledge-graph/`

**特点**：
- GraphRAG实现
- 多跳推理
- 路径可视化

### 案例5：多模态产品问答

**路径**: `projects/case5-multimodal/`

**特点**：
- 图文混合检索
- CLIP/GPT-4V支持

### 案例6：企业级RAG平台

**路径**: `projects/case6-enterprise-platform/`

**特点**：
- FastAPI RESTful API
- JWT认证
- Redis缓存
- 用户权限管理

---

## 🛠️ 技术栈

### 核心框架
- **LangChain**: 强大的LLM应用开发框架，支持链式调用、Agent等
- **LlamaIndex**: 专注于数据索引和检索的RAG框架

### 大语言模型（LLM）
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **本地模型**: 支持通过Ollama使用Llama 3、Qwen等开源模型

### 向量数据库
- **ChromaDB**: 轻量级本地向量数据库
- **Pinecone**: 全托管向量数据库服务
- **MongoDB Atlas Vector Search**: MongoDB原生向量搜索
- **Weaviate**: 开源向量搜索引擎

### 嵌入模型
- **OpenAI Embeddings**: text-embedding-3-small, text-embedding-3-large
- **HuggingFace**: sentence-transformers系列（all-MiniLM-L6-v2, m3e-base等）
- **FlagEmbedding**: 中文优化嵌入模型（bge系列）

### RAG优化技术
- **混合检索**: Vector Search + BM25关键词检索
- **重排序**: CrossEncoder、Cohere Rerank
- **查询增强**: HyDE（假设文档嵌入）、Query Rewriting、Query Expansion
- **高级分块**: Semantic Chunking、Recursive Character Splitting
- **检索压缩**: Context Compression、LLMContextualCompression

### Agent架构
- **ReAct Agent**: 推理+行动模式
- **Self-RAG**: 自我反思RAG
- **CRAG**: 校正RAG
- **Agentic RAG**: Agent驱动的动态检索
- **Deep Research Agent**: 多轮深度推理Agent

### 知识图谱
- **GraphRAG**: 结合知识图谱的RAG
- **Neo4j**: 图数据库存储
- **NetworkX**: 图计算和分析

### Web框架
- **Streamlit**: 快速构建交互式界面
- **FastAPI**: 高性能异步API框架
- **Jupyter**: 交互式Notebook环境

### 部署运维
- **Docker**: 容器化部署
- **Kubernetes**: 容器编排
- **GitHub Actions**: CI/CD自动化
- **Prometheus + Grafana**: 监控和告警

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

### 学习教程

#### 选项1：在线阅读（推荐）

访问 [GitHub Pages](https://vivy-yi.github.io/rag-tutorial/)

#### 选项2：本地阅读

```bash
# 查看教程文档
cd docs/01-基础入门
open 01-RAG技术概述.md
```

#### 选项3：Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook notebooks/

# 浏览到对应模块的notebook目录
# 例如：notebooks/module1/
```

#### 选项4：运行实战案例

```bash
# 例如运行案例1
cd projects/case1-customer-service
python main.py
```

---

## 📚 学习路径

### 路径1：快速入门 (2-3周)

```
docs/01-基础入门 → projects/case1-customer-service → docs/04-生产部署
```

### 路径2：系统学习 (6-8周)

```
docs/01-基础入门 → docs/02-核心优化 → docs/03-高级架构 → projects/case1-2-3
```

### 路径3：专家级 (10-12周)

```
所有docs模块 → 所有projects → 深入优化 → docs/04-生产部署
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
├── docs/                       # 教程文档（按模块）
│   ├── 01-基础入门/
│   │   ├── *.md
│   │   ├── notebooks/
│   │   └── exercises/
│   ├── 02-核心优化/
│   ├── 03-高级架构/
│   └── 04-生产部署/
│
├── notebooks/                  # Jupyter Notebooks（按模块）
│   ├── module1/
│   ├── module2/
│   ├── module3/
│   └── module4/
│
├── exercises/                   # 练习题（按模块）
│   ├── module1/
│   ├── module2/
│   ├── module3/
│   └── module4/
│
├── projects/                    # 实战案例
│   ├── case1-customer-service/
│   ├── case2-doc-qa/
│   ├── case3-research-agent/
│   ├── case4-knowledge-graph/
│   ├── case5-multimodal/
│   └── case6-enterprise-platform/
│
├── assets/                      # 图片资源
│   └── images/
│       ├── module1-basic/
│       ├── module2-optimization/
│       ├── module3-advanced/
│       ├── module4-production/
│       └── logos/
│
├── README.md                    # 本文件
├── CONTRIBUTING.md              # 贡献指南
├── .gitignore                   # Git忽略配置
├── requirements.txt             # Python依赖
└── LICENSE                      # MIT许可证
```

---

## 🎓 学习建议

### 循序渐进

1. **先掌握基础**：完成docs/01-基础入门
2. **学习优化技术**：学习docs/02-核心优化
3. **实践高级架构**：探索docs/03-高级架构
4. **生产部署**：了解docs/04-生产部署

### 理论与实践结合

- 每章都有对应的Jupyter Notebook
- 边学边练，完成exercises/中的练习题
- 运行projects/中的实战案例

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
3. 确保你的代码符合我们的代码规范
4. 提交代码 (`git commit -m 'Add some AmazingFeature'`)
5. 推送到分支 (`git push origin feature/AmazingFeature`)
6. 开启 Pull Request

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
- **Email**: xiaoluopupu@gmail.com

---

## 🔖 关键词

RAG、检索增强生成、Retrieval-Augmented Generation、LangChain、LlamaIndex、向量数据库、Vector Database、大语言模型、LLM、GPT-4、Claude、ChromaDB、Pinecone、Agent、ReAct、Self-RAG、GraphRAG、HyDE、混合检索、重排序、嵌入模型、Embedding、OpenAI、中文教程、人工智能教程、AI应用开发、知识库问答、智能客服

---

## ⭐ 如果这个教程对你有帮助

请给个Star支持一下！🙏

[![GitHub stars](https://img.shields.io/github/stars/vivy-yi/rag-tutorial?style=social)](https://github.com/vivy-yi/rag-tutorial/stargazers)
