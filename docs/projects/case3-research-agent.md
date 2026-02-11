# 案例3：AI研究助手Agent

> **难度**: ⭐⭐⭐ 高级 | **技术栈**: LangChain Agent, ReAct, ArXiv API, Google Scholar

使用ReAct Agent模式构建能够自动搜索论文、生成报告的AI研究助手

---

## 🎯 案例特点

- ✅ **ReAct Agent**: 推理+行动模式
- ✅ **多工具协作**: ArXiv搜索、论文摘要、网页浏览
- ✅ **自动报告**: 生成结构化研究报告
- ✅ **迭代推理**: 任务分解和逐步执行

---

## 🚀 快速开始

```bash
cd projects/case3-research-agent
pip install -r requirements.txt
python main.py
```

---

## 📁 项目结构

```
case3-research-agent/
├── main.py              # 主程序
├── research_agent.py    # Agent核心实现
├── tools.py             # 工具集合
└── requirements.txt
```

---

## 🔑 核心代码

### ReAct Agent实现

```python
# research_agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

def create_research_agent():
    # 定义工具
    tools = [
        Tool(
            name="arxiv_search",
            func=search_arxiv,
            description="搜索学术论文，输入关键词返回相关论文列表"
        ),
        Tool(
            name="paper_summary",
            func=summarize_paper,
            description="获取论文摘要和关键信息，输入论文ID"
        ),
        Tool(
            name="google_scholar",
            func=search_scholar,
            description="在Google Scholar上搜索相关研究"
        )
    ]

    # 创建ReAct agent
    agent = create_react_agent(
        llm=ChatOpenAI(temperature=0),
        tools=tools,
        prompt=react_prompt_template
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )
```

### 任务执行流程

```python
def research_topic(topic: str):
    """研究一个主题并生成报告"""
    agent = create_research_agent()

    # 多步推理
    steps = [
        f"搜索关于'{topic}'的最新论文",
        "分析这些论文的摘要",
        "总结主要研究发现",
        "生成结构化报告"
    ]

    results = []
    for step in steps:
        result = agent.run(step)
        results.append(result)

    return generate_report(results)
```

---

## 🎓 学习要点

1. **Agent架构**
   - ReAct模式
   - 工具定义和使用
   - 推理链构建

2. **多步推理**
   - 任务分解
   - 上下文维护
   - 迭代优化

3. **API集成**
   - ArXiv API
   - Google Scholar
   - 论文解析

---

## 📊 示例输出

**用户**: 帮我研究"GraphRAG"的最新进展

**Agent执行流程**:
1. Thought: 需要搜索GraphRAG相关论文
2. Action: arxiv_search("GraphRAG")
3. Observation: 找到15篇相关论文...
4. Thought: 分析这些论文的摘要
5. Action: paper_summary(paper_ids)
6. Observation: 论文主要关注...
7. Final Answer: 生成完整报告

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case3-research-agent)**

**[← 返回案例列表](index.md)**
