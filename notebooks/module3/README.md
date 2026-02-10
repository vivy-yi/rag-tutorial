# 模块3 Jupyter Notebooks

> 动手实践高级架构模式：Agent、知识图谱、多模态RAG

---

## 📚 Notebooks概览

本目录包含3个交互式Jupyter Notebook，涵盖模块3的核心技术点：

| Notebook | 主题 | 难度 | 预计时间 |
|----------|------|------|----------|
| 13_react_agent.ipynb | ReAct Agent实现 | ⭐⭐⭐ | 90分钟 |
| 15_graph_rag.ipynb | 知识图谱RAG | ⭐⭐⭐⭐ | 110分钟 |
| 16_multimodal_rag.ipynb | 多模态RAG系统 | ⭐⭐⭐⭐⭐ | 120分钟 |

---

## 🔧 环境配置

### 基础环境

```bash
# Python 3.9+
python --version

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 依赖安装

```bash
# 安装核心依赖
pip install jupyter networkx matplotlib sentence-transformers

# 安装额外依赖
pip install spacy transformers torch openai

# 下载中文模型（用于NER）
python -m spacy download zh_core_web_sm

# 启动Jupyter
jupyter notebook
```

### requirements.txt

```
jupyter>=1.0.0
networkx>=3.0
matplotlib>=3.5.0
sentence-transformers>=2.2.0
spacy>=3.5.0
transformers>=4.30.0
torch>=2.0.0
openai>=1.0.0
pillow>=9.0.0
```

---

## 📖 Notebooks详细说明

### 1. ReAct Agent实现 (13_react_agent.ipynb)

**学习目标**：
- 理解ReAct模式原理
- 实现Thought-Action-Observation循环
- 构建可扩展的工具系统
- 实现记忆机制

**主要内容**：
1. **环境准备**：配置必要的库
2. **工具系统**：实现Search、Calculator、RAG工具
3. **ReAct Agent**：完整实现核心循环
4. **运行实验**：测试不同查询类型
5. **高级功能**：添加记忆、可视化
6. **性能评估**：统计迭代次数和成功率

**核心代码**：
```python
class ReActAgent:
    def run(self, query: str, verbose: bool = True) -> Dict:
        for iteration in range(self.max_iterations):
            # 1. 构建提示
            prompt = self._build_prompt(query, steps)

            # 2. LLM生成思考和行动
            thought, action, action_input = self._parse_response(llm_response)

            # 3. 执行工具
            observation = self.tools[action](action_input)

            # 4. 检查是否完成
            if action == "Finish":
                return answer
```

**实验输出**：
- 完整的推理轨迹
- 工具调用序列
- 可视化执行过程

---

### 2. 知识图谱RAG (15_graph_rag.ipynb)

**学习目标**：
- 构建知识图谱数据结构
- 实现实体识别和关系抽取
- 执行多跳推理检索
- 构建完整GraphRAG系统

**主要内容**：
1. **图谱构建**：实体、关系定义
2. **实体识别**：从查询中提取实体
3. **图谱检索**：多跳推理算法
4. **GraphRAG系统**：完整实现
5. **可视化**：图谱和推理路径
6. **性能评估**：不同跳数对比

**核心代码**：
```python
class GraphRAG:
    def query(self, query: str, max_hops: int = 2) -> Dict:
        # 1. 实体识别
        entities = self.entity_extractor.extract_entities(query)

        # 2. 图谱检索
        graph_result = self.graph_retriever.retrieve_by_query(
            query, max_hops=max_hops
        )

        # 3. 构建图谱上下文
        graph_context = self._build_graph_context(graph_result)

        # 4. 生成答案
        answer = self._generate_answer(query, graph_context)
```

**实验输出**：
- 知识图谱可视化
- 推理路径展示
- 子图提取结果
- 性能对比报告

---

### 3. 多模态RAG系统 (16_multimodal_rag.ipynb)

**学习目标**：
- 使用CLIP进行跨模态嵌入
- 实现图文混合检索
- 构建多模态问答系统
- 创建Web界面

**主要内容**：
1. **CLIP模型**：图像-文本嵌入
2. **跨模态检索**：图像+文本搜索
3. **结果融合**：RRF策略
4. **多模态Agent**：GPT-4V集成
5. **Web界面**：Streamlit实现
6. **评估分析**：Recall@K指标

**核心代码**：
```python
class MultiModalRAG:
    def retrieve_multimodal(self, query_text: str,
                          query_image: str = None) -> List[Dict]:
        results = []

        # 文本检索
        text_results = self.retrieve_by_text(query_text)
        results.extend(text_results)

        # 图像检索（如果提供）
        if query_image:
            image_results = self.retrieve_by_image(query_image)
            results.extend(image_results)

        # 融合排序
        return self._fuse_and_rerank(results)
```

**实验输出**：
- 图像检索结果
- 图文混合查询结果
- 可视化界面
- 性能评估报告

---

## 🚀 使用指南

### 快速开始

1. **克隆或下载教程**：
```bash
cd RAG完整教程/03-高级架构/notebooks
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **启动Jupyter**：
```bash
jupyter notebook
```

4. **选择Notebook**：按顺序或根据兴趣选择

### 推荐学习路径

**路径1：系统学习**
```
13_react_agent → 15_graph_rag → 16_multimodal_rag
```
适合：希望全面掌握所有技术

**路径2：重点突破**
```
15_graph_rag → 13_react_agent → 16_multimodal_rag
```
适合：对知识图谱特别感兴趣

**路径3：快速上手**
```
13_react_agent → 16_multimodal_rag
```
适合：时间有限，想学最实用的技术

---

## 💡 实验建议

### ReAct Agent实验

1. **基础实验**：
   - 运行预定义查询
   - 观察推理轨迹
   - 理解工具调用机制

2. **扩展实验**：
   - 添加新工具（Weather、News等）
   - 优化提示词
   - 实现多轮对话

3. **进阶实验**：
   - 集成真实LLM（OpenAI）
   - 实现任务拆分
   - 添加反思机制

### 知识图谱实验

1. **基础实验**：
   - 构建小型领域图谱
   - 测试多跳推理
   - 可视化结果

2. **扩展实验**：
   - 使用spaCy进行NER
   - 实现实体消歧
   - 添加关系权重

3. **进阶实验**：
   - 融合向量检索
   - 实现子图嵌入
   - 优化路径排序

### 多模态RAG实验

1. **基础实验**：
   - 准备图像数据集
   - 测试CLIP嵌入
   - 实现图像检索

2. **扩展实验**：
   - 图文混合检索
   - 结果融合优化
   - 构建Web界面

3. **进阶实验**：
   - 集成GPT-4V
   - 实现多模态Agent
   - 性能调优

---

## 📊 预期学习成果

完成所有Notebooks后，你将能够：

### ✅ 技术能力

1. **Agent开发**
   - 设计和实现ReAct Agent
   - 创建可扩展的工具系统
   - 处理复杂推理任务

2. **知识图谱**
   - 构建领域知识图谱
   - 实现多跳推理
   - 开发GraphRAG系统

3. **多模态系统**
   - 使用CLIP等模型
   - 实现跨模态检索
   - 构建多模态应用

### ✅ 项目经验

- 完整的ReAct Agent实现
- 端到端的知识图谱系统
- 生产级的多模态RAG

### ✅ 代码能力

- 清晰的代码结构
- 良好的模块化设计
- 完整的文档和注释

---

## 🐛 常见问题

### Q1: Jupyter无法启动？

**解决方案**：
```bash
# 检查Jupyter是否安装
jupyter --version

# 重新安装
pip install --upgrade jupyter

# 检查端口占用
lsof -i :8888
```

### Q2: 依赖安装失败？

**解决方案**：
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name

# 分步安装
pip install networkx matplotlib
pip install sentence-transformers
```

### Q3: 图表不显示中文？

**解决方案**：
```python
# 安装中文字体
# Mac: 系统自带 Arial Unicode MS
# Linux: sudo apt-get install fonts-wqy-zenhei
# Windows: 系统自带 SimHei

# 在Notebook中设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
```

### Q4: CLIP模型下载慢？

**解决方案**：
```python
# 使用国内镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或使用sentence-transformers
model = SentenceTransformer('clip-ViT-B-32')
```

### Q5: 内存不足？

**解决方案**：
```python
# 使用较小的模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 而不是大型模型

# 批量处理
batch_size = 8  # 减小批量大小
```

---

## 📚 扩展资源

### 官方文档

- [NetworkX文档](https://networkx.org/documentation/stable/)
- [LangChain文档](https://python.langchain.com/docs/get_started/introduction)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs/)

### 推荐阅读

- "ReAct: Synergizing Reasoning and Acting in Language Models" (原论文)
- "GraphRAG: Boosting RAG with Knowledge Graphs"
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP论文)

### 社区资源

- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Sentence-Transformers GitHub](https://github.com/UKPLab/sentence-transformers)

---

## 🎯 学习检查清单

使用此清单跟踪学习进度：

### ReAct Agent
- [ ] 理解ReAct模式
- [ ] 实现基础工具系统
- [ ] 完成Agent类
- [ ] 运行实验并观察输出
- [ ] 添加自定义工具
- [ ] 实现记忆功能

### 知识图谱RAG
- [ ] 构建知识图谱
- [ ] 实现实体识别
- [ ] 完成多跳检索
- [ ] 构建GraphRAG系统
- [ ] 可视化图谱和路径
- [ ] 性能评估

### 多模态RAG
- [ ] 使用CLIP模型
- [ ] 实现图像检索
- [ ] 完成图文融合
- [ ] 构建多模态Agent
- [ ] 创建Web界面
- [ ] 评估系统效果

---

## 🤝 贡献指南

欢迎改进这些Notebooks！

1. Fork项目
2. 创建改进分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交Pull Request

---

## 📝 许可证

MIT License

---

## 🙏 致谢

- NetworkX团队
- LangChain团队
- Sentence-Transformers团队
- OpenAI团队
- 所有贡献者

---

**最后更新**：2025-02-10
**版本**：v1.0.0

**祝你学习愉快！** 🚀
