# 实战案例概览

本教程包含 **6个完整的实战案例**，涵盖从基础到企业级的各种RAG应用场景。

## 📦 案例列表

| 案例 | 技术栈 | 难度 | 说明 |
|------|--------|------|------|
| [案例1：智能客服RAG系统](case1-customer-service.md) | LangChain, OpenAI | ⭐ 入门 | 基础RAG应用，支持多轮对话 |
| [案例2：技术文档问答系统](case2-doc-qa.md) | LangChain, Hybrid Search | ⭐⭐ 进阶 | 混合检索 + 重排序 |
| [案例3：AI研究助手Agent](case3-research-agent.md) | LangChain Agent, ArXiv | ⭐⭐⭐ 高级 | ReAct Agent模式 |
| [案例4：企业知识图谱问答](case4-knowledge-graph.md) | GraphRAG, Neo4j | ⭐⭐⭐ 高级 | 多跳推理，路径可视化 |
| [案例5：多模态产品问答](case5-multimodal.md) | CLIP, GPT-4V | ⭐⭐⭐ 高级 | 图文混合检索 |
| [案例6：企业级RAG平台](case6-enterprise-platform.md) | FastAPI, JWT, Redis | ⭐⭐⭐⭐ 专家 | RESTful API，完整权限 |

## 🎯 学习路径

### 入门路径（适合初学者）
```
案例1 → 案例2 → 案例3
```

### 进阶路径（有经验开发者）
```
案例2 → 案例4 → 案例5
```

### 专家路径（企业级开发）
```
案例3 → 案例4 → 案例5 → 案例6
```

## 💡 使用建议

1. **边学边练**：每个案例都有完整的代码，建议clone后本地运行
2. **循序渐进**：从简单到复杂，逐步掌握RAG技术
3. **实际应用**：结合自己的业务场景进行改造
4. **阅读源码**：深入理解实现细节和最佳实践

## 📂 案例源码

所有案例源码托管在 GitHub：
```bash
git clone https://github.com/vivy-yi/rag-tutorial.git
cd rag-tutorial/projects
```

每个案例都包含：
- 📄 `README.md` - 案例说明文档
- 🐍 `main.py` - 主程序入口
- 🔧 `*.py` - 核心实现代码
- 📦 `requirements.txt` - Python依赖

---

**下一步**：选择一个案例开始学习吧！🚀
