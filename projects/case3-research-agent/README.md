# 案例3：AI研究助手Agent

> 使用ReAct Agent构建能够自主搜索、分析和总结的AI研究助手

---

## 📋 案例概述

### 业务场景

研究人员和开发者面临信息过载：
- ✗ 论文数量爆炸式增长
- ✗ 技术更新迭代快
- ✗ 跨领域知识整合困难
- ✗ 代码实现耗时

### Agent解决方案

构建自主研究助手Agent：
- ✅ 自动文献检索
- ✅ 技术对比分析
- ✅ 代码示例生成
- ✅ 研究报告撰写
- ✅ 多步推理规划

---

## 🎯 功能需求

### 核心功能

1. **文献搜索**
   - ArXiv论文检索
   - Google Scholar搜索
   - GitHub代码仓库
   - Stack Overflow问答

2. **内容分析**
   - 论文摘要提取
   - 核心方法总结
   - 实验结果分析
   - 代码实现理解

3. **对比研究**
   - 多模型性能对比
   - 优缺点分析
   - 适用场景评估
   - 发展趋势预测

4. **报告生成**
   - 结构化研究报告
   - Markdown格式
   - 包含引用链接
   - 代码示例

### Agent能力

- **自主规划**：分解复杂任务
- **工具调用**：使用多种API
- **多步推理**：链式思考
- **自我反思**：验证和改进
- **记忆管理**：保持上下文

---

## 🏗️ 系统架构

### Agent架构

```
┌─────────────────────────────────────────┐
│         研究助手Agent                    │
│  ┌───────────────────────────────────┐  │
│  │      Planner (规划器)              │  │
│  │  - 任务分解                         │  │
│  │  - 步骤规划                         │  │
│  │  - 动态调整                         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │      ReAct Loop (推理循环)         │  │
│  │  - Thought                         │  │
│  │  - Action                          │  │
│  │  - Observation                     │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │      Tools (工具集)                │  │
│  │  - ArXivSearch                    │  │
│  │  - GitHubSearch                   │  │
│  │  - PaperAnalysis                  │  │
│  │  - CodeGeneration                 │  │
│  │  - ReportWriting                  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │      Memory (记忆)                 │  │
│  │  - 短期记忆 (当前会话)             │  │
│  │  - 长期记忆 (向量存储)             │  │
│  │  - 知识库 (重要发现)               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 技术栈

**Agent框架**：
- LangChain Agents
- LangGraph (工作流)

**工具**：
- ArXiv API
- GitHub API
- Google Scholar (爬虫)
- OpenAI GPT-4

**存储**：
- ChromaDB (记忆)
- PostgreSQL (会话)

**前端**：
- Streamlit

---

## 💻 核心实现

### 1. 工具定义

```python
# tools/research_tools.py
import requests
import arxiv
from typing import List, Dict
import re

class ArXivSearchTool:
    """ArXiv论文搜索工具"""

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        搜索ArXiv论文

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            论文列表
        """
        # 使用arxiv库
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for result in search.results():
            paper = {
                'title': result.title,
                'authors': [a.name for a in result.authors],
                'summary': result.summary.replace('\n', ' '),
                'published': result.published.strftime('%Y-%m-%d'),
                'url': result.entry_id,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category
            }
            results.append(paper)

        return results

    def get_paper_details(self, paper_id: str) -> Dict:
        """获取论文详情"""
        search = arxiv.Search(id_list=[paper_id])
        result = next(search.results())

        return {
            'title': result.title,
            'abstract': result.summary,
            'authors': [a.name for a in result.authors],
            'categories': result.categories,
            'pdf_url': result.pdf_url
        }


class GitHubSearchTool:
    """GitHub代码搜索工具"""

    def __init__(self, token: str = None):
        self.token = token
        self.base_url = "https://api.github.com"

    def search_repositories(self,
                           query: str,
                           sort: str = "stars",
                           per_page: int = 10) -> List[Dict]:
        """
        搜索GitHub仓库

        Args:
            query: 搜索查询
            sort: 排序方式 (stars, forks, updated)
            per_page: 每页结果数

        Returns:
            仓库列表
        """
        url = f"{self.base_url}/search/repositories"
        params = {
            'q': query,
            'sort': sort,
            'per_page': per_page
        }

        headers = {}
        if self.token:
            headers['Authorization'] = f"token {self.token}"

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data['items']:
            repo = {
                'name': item['name'],
                'full_name': item['full_name'],
                'description': item['description'],
                'url': item['html_url'],
                'stars': item['stargazers_count'],
                'language': item['language'],
                'updated_at': item['updated_at']
            }
            results.append(repo)

        return results

    def search_code(self,
                   query: str,
                   language: str = None) -> List[Dict]:
        """
        搜索代码

        Args:
            query: 代码查询
            language: 编程语言

        Returns:
            代码片段列表
        """
        url = f"{self.base_url}/search/code"
        q = query
        if language:
            q += f" language:{language}"

        params = {'q': q, 'per_page': 10}
        headers = {}
        if self.token:
            headers['Authorization'] = f"token {self.token}"

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        return data.get('items', [])


class PaperAnalysisTool:
    """论文分析工具"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_key_points(self, paper: Dict) -> Dict:
        """
        提取论文关键点

        Args:
            paper: 论文信息

        Returns:
            关键点提取结果
        """
        prompt = f"""分析以下论文，提取关键信息：

标题: {paper['title']}
摘要: {paper['summary']}

请提取：
1. 研究问题
2. 核心方法
3. 主要贡献
4. 实验结果

以JSON格式返回。"""

        response = self.llm.generate(prompt)

        # 解析LLM响应（实际需要更robust的解析）
        try:
            import json
            return json.loads(response)
        except:
            return {
                'research_question': '待提取',
                'core_method': '待提取',
                'contributions': '待提取',
                'results': '待提取'
            }

    def compare_papers(self,
                      papers: List[Dict]) -> str:
        """
        对比多篇论文

        Args:
            papers: 论文列表

        Returns:
            对比分析文本
        """
        prompt = f"""对比以下{len(papers)}篇论文：

"""
        for i, paper in enumerate(papers, 1):
            prompt += f"\n论文{i}: {paper['title']}\n"
            prompt += f"摘要: {paper['summary'][:200]}...\n"

        prompt += """
请从以下方面进行对比：
1. 方法对比
2. 性能对比
3. 优缺点分析
4. 适用场景

生成结构化的对比分析。"""

        return self.llm.generate(prompt)


class CodeGenerationTool:
    """代码生成工具"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def implement_paper(self,
                       paper: Dict,
                       language: str = "Python") -> str:
        """
        根据论文实现代码

        Args:
            paper: 论文信息
            language: 编程语言

        Returns:
            生成的代码
        """
        prompt = f"""基于以下论文生成{language}代码实现：

论文标题: {paper['title']}
核心方法: {paper['summary'][:500]}

要求：
1. 实现核心算法
2. 添加详细注释
3. 包含使用示例
4. 考虑边界情况

生成完整可运行的代码。"""

        return self.llm.generate(prompt)

    def explain_code(self,
                    code: str,
                    language: str = "Python") -> str:
        """
        解释代码

        Args:
            code: 代码片段
            language: 编程语言

        Returns:
            代码解释
        """
        prompt = f"""详细解释以下{language}代码：

```{language}
{code}
```

请说明：
1. 代码功能
2. 算法原理
3. 关键步骤
4. 时间复杂度
5. 可能的优化"""

        return self.llm.generate(prompt)
```

### 2. ReAct Agent实现

```python
# agent/research_agent.py
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain import hub
from typing import List, Dict, Optional
import json

class ResearchAgent:
    """
    AI研究助手Agent

    使用ReAct模式自主完成研究任务
    """

    def __init__(self,
                 openai_api_key: str,
                 github_token: str = None):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=openai_api_key
        )

        # 初始化工具
        self.tools = self._create_tools(github_token)

        # 创建Agent
        self.agent = self._create_agent()

        # 记忆
        self.memory = []

    def _create_tools(self, github_token: str) -> List[Tool]:
        """创建工具集"""

        # ArXiv搜索
        arxiv_tool = ArXivSearchTool()

        # GitHub搜索
        github_tool = GitHubSearchTool(token=github_token)

        # 论文分析
        paper_analysis = PaperAnalysisTool(self.llm)

        # 代码生成
        code_gen = CodeGenerationTool(self.llm)

        tools = [
            Tool(
                name="ArXivSearch",
                func=lambda q: json.dumps(
                    arxiv_tool.search(q, max_results=5),
                    ensure_ascii=False,
                    indent=2
                ),
                description="""搜索ArXiv学术论文。
                输入：搜索查询，例如'transformer architecture'
                输出：相关论文列表（包含标题、作者、摘要、链接）"""
            ),
            Tool(
                name="GitHubSearch",
                func=lambda q: json.dumps(
                    github_tool.search_repositories(q),
                    ensure_ascii=False,
                    indent=2
                ),
                description="""搜索GitHub代码仓库。
                输入：搜索查询，例如'RAG implementation'
                输出：相关仓库列表（包含名称、描述、星标数、链接）"""
            ),
            Tool(
                name="AnalyzePaper",
                func=lambda p: json.dumps(
                    paper_analysis.extract_key_points(json.loads(p)),
                    ensure_ascii=False
                ),
                description="""分析论文关键点。
                输入：论文信息JSON（包含标题和摘要）
                输出：关键点提取（研究问题、核心方法、主要贡献、实验结果）"""
            ),
            Tool(
                name="GenerateCode",
                func=lambda x: code_gen.implement_paper(json.loads(x)),
                description="""根据论文生成代码实现。
                输入：论文信息JSON
                输出：完整代码实现（带注释）"""
            ),
            Tool(
                name="ComparePapers",
                func=lambda p: paper_analysis.compare_papers(json.loads(p)),
                description="""对比多篇论文。
                输入：论文列表JSON
                输出：结构化对比分析"""
            )
        ]

        return tools

    def _create_agent(self) -> AgentExecutor:
        """创建Agent"""

        # 获取prompt模板
        prompt = hub.pull("hwchase17/openai-tools-agent")

        # 创建agent
        agent = create_openai_tools_agent(
            self.llm,
            self.tools,
            prompt
        )

        # 创建executor
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

        return executor

    def research(self,
                query: str,
                save_report: bool = True) -> Dict:
        """
        执行研究任务

        Args:
            query: 研究查询
            save_report: 是否保存报告

        Returns:
            研究结果
        """
        print(f"🔬 开始研究任务: {query}\n")

        # 执行agent
        try:
            result = self.agent.invoke({"input": query})

            # 保存到记忆
            self.memory.append({
                'query': query,
                'result': result['output'],
                'steps': len(result.get('intermediate_steps', []))
            })

            # 生成报告
            if save_report:
                report = self._generate_report(query, result)
                return {
                    'answer': result['output'],
                    'report': report,
                    'steps': result.get('intermediate_steps', [])
                }

            return {
                'answer': result['output'],
                'steps': result.get('intermediate_steps', [])
            }

        except Exception as e:
            return {
                'error': str(e),
                'answer': f"研究过程中发生错误: {str(e)}"
            }

    def _generate_report(self,
                        query: str,
                        result: Dict) -> str:
        """生成研究报告"""

        report = f"""# 研究报告

## 研究问题
{query}

## 研究过程
"""

        # 添加步骤
        steps = result.get('intermediate_steps', [])
        for i, step in enumerate(steps, 1):
            action, observation = step
            report += f"\n### 步骤{i}: {action.tool}\n"
            report += f"{action.tool_input}\n\n"
            report += f"**结果**: {observation[:200]}...\n\n"

        # 添加结论
        report += f"\n## 结论\n\n{result['output']}\n"

        return report
```

### 3. Streamlit界面

```python
# app.py
import streamlit as st
from agent.research_agent import ResearchAgent
import json

st.set_page_config(
    page_title="AI研究助手",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI研究助手Agent")
st.markdown("自主搜索、分析和总结技术文献")

# 侧边栏
with st.sidebar:
    st.header("配置")

    api_key = st.text_input("OpenAI API Key", type="password")
    github_token = st.text_input("GitHub Token (可选)", type="password")

    st.divider()

    st.subheader("研究模板")
    templates = [
        "对比不同RAG优化方法",
        "查找Transformer最新进展",
        "分析多模态学习技术",
        "研究图神经网络应用"
    ]

    for template in templates:
        if st.button(template, key=template):
            st.session_state.query = template

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area(
        "研究问题",
        placeholder="例如：对比RAG的不同优化方法，包括混合检索、重排序、查询优化等",
        height=100,
        key="query_input"
    )

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        research_button = st.button("🚀 开始研究", type="primary")

    with col_btn2:
        if st.button("💾 查看历史"):
            st.session_state.show_history = not st.session_state.get('show_history', False)

# 显示历史
if st.session_state.get('show_history', False):
    st.subheader("📚 研究历史")
    if 'agent' in st.session_state and st.session_state.agent.memory:
        for i, item in enumerate(st.session_state.agent.memory[-5:], 1):
            with st.expander(f"{i}. {item['query']}", expanded=False):
                st.write(f"**步骤数**: {item['steps']}")
                st.write(f"**结果**:\n{item['answer'][:300]}...")
    else:
        st.info("暂无研究历史")

# 执行研究
if research_button and query:
    if not api_key:
        st.error("请输入OpenAI API Key")
    else:
        # 初始化agent
        if 'agent' not in st.session_state or st.session_state.get('last_api_key') != api_key:
            st.session_state.agent = ResearchAgent(
                openai_api_key=api_key,
                github_token=github_token or None
            )
            st.session_state.last_api_key = api_key

        # 创建进度容器
        progress_container = st.container()

        with progress_container:
            st.info("🤖 Agent正在研究中...")

            # 执行研究
            result = st.session_state.agent.research(query)

            # 显示结果
            if 'error' in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ 研究完成！")

                # 显示答案
                st.subheader("📝 研究结论")
                st.write(result['answer'])

                # 显示步骤
                if 'steps' in result and result['steps']:
                    with st.expander("🔍 查看研究步骤", expanded=False):
                        for i, (action, observation) in enumerate(result['steps'], 1):
                            st.markdown(f"**步骤{i}**: `{action.tool}`")
                            st.code(action.tool_input, language="text")
                            st.text(observation[:500] + "..." if len(observation) > 500 else observation)
                            st.divider()

                # 下载报告
                if 'report' in result:
                    st.download_button(
                        label="📥 下载研究报告",
                        data=result['report'],
                        file_name=f"research_report_{hash(query)}.md",
                        mime="text/markdown"
                    )
```

---

## 📊 使用示例

### 示例1：技术对比研究

```python
# 示例：研究查询
query = """
对比RAG的不同优化方法，包括：
1. 混合检索（Vector + BM25）
2. 重排序（CrossEncoder）
3. 查询优化（HyDE, Query Expansion）
4. 智能分块

请分析每种方法的原理、优缺点、适用场景和性能提升。
"""

# 执行研究
agent = ResearchAgent(openai_api_key="your-key")
result = agent.research(query)

print(result['answer'])
```

### 示例2：论文追踪

```python
query = """
查找2024年大模型推理优化技术的最新论文，重点关注：
1. 推理加速方法
2. 量化技术
3. KV Cache优化
4. 并行策略

请总结核心方法和性能提升。
"""

result = agent.research(query)
```

### 示例3：代码实现

```python
query = """
研究GraphRAG技术，并：
1. 查找相关论文
2. 搜索GitHub实现
3. 生成Python代码示例
4. 分析与传统RAG的优劣势
"""

result = agent.research(query)
```

---

## 🧪 评估

### Agent性能评估

```python
# evaluation.py
class AgentEvaluator:
    """Agent评估器"""

    def __init__(self, agent):
        self.agent = agent

    def evaluate_research_quality(self,
                                 query: str,
                                 ground_truth: Dict) -> Dict:
        """
        评估研究质量

        Args:
            query: 研究查询
            ground_truth: 标准答案

        Returns:
            评估指标
        """
        result = self.agent.research(query, save_report=False)

        metrics = {}

        # 1. 完整性（是否覆盖关键点）
        required_points = ground_truth.get('key_points', [])
        covered_points = sum(
            1 for point in required_points
            if point.lower() in result['answer'].lower()
        )
        metrics['completeness'] = covered_points / len(required_points) if required_points else 0

        # 2. 准确性（与标准答案的一致性）
        # 这里可以使用LLM来评估
        metrics['accuracy'] = self._llm_evaluate_accuracy(
            result['answer'],
            ground_truth.get('answer', '')
        )

        # 3. 工具使用效率
        steps = result.get('steps', [])
        metrics['tool_efficiency'] = len(steps)

        # 4. 推理深度
        metrics['reasoning_depth'] = self._analyze_reasoning_depth(steps)

        return metrics

    def _llm_evaluate_accuracy(self,
                              generated: str,
                              reference: str) -> float:
        """使用LLM评估准确性"""
        prompt = f"""评估以下答案的准确性（0-1分）：

生成答案: {generated}

参考答案: {reference}

请给出0-1之间的分数，保留两位小数。只返回分数。"""

        response = self.agent.llm.predict(prompt)

        try:
            return float(response.strip())
        except:
            return 0.5  # 默认分数
```

---

## 🎓 学习要点

完成本案例后，你将掌握：

### ✅ Agent开发
- ReAct模式实现
- 工具定义和集成
- 多步推理规划
- 自主任务执行

### ✅ API集成
- ArXiv API
- GitHub API
- LLM API调用
- 错误处理

### ✅ 研究流程
- 文献检索
- 内容分析
- 对比研究
- 报告生成

### ✅ 系统设计
- Agent架构设计
- 工具抽象
- 记忆管理
- 性能优化

---

## 🚀 进阶方向

1. **高级工具**
   - Web爬虫（Google Scholar）
   - PDF解析（arXiv PDF）
   - 代码执行（沙箱）
   - 可视化生成

2. **多Agent协作**
   - 专门搜索Agent
   - 分析Agent
   - 写作Agent
   - Manager协调

3. **知识库构建**
   - 向量化论文
   - 语义检索
   - 引用网络
   - 趋势分析

---

## 📚 参考资源

- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ArXiv API](http://export.arxiv.org/api_help/)
- [GitHub API](https://docs.github.com/en/rest)

---

**开始构建你的AI研究助手吧！** 🚀
