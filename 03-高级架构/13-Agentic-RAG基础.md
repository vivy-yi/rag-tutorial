# 第13章：Agentic RAG基础

> 让RAG系统拥有"大脑"：Agent架构让系统自主思考、规划和决策，解决传统RAG无法处理的复杂问题！

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解Agent的核心概念和原理
- [ ] 掌握ReAct（推理+行动）模式
- [ ] 实现工具调用机制
- [ ] 构建智能问答Agent
- [ ] 完成第一个Agent项目

**预计学习时间**：6小时
**难度等级**：⭐⭐⭐⭐☆

---

## 前置知识

- [ ] 完成模块1和模块2
- [ ] 熟悉Python异步编程
- [ ] 了解LangChain或LlamaIndex基础
- [ ] 理解Prompt Engineering

---

## 13.1 什么是Agent？

### 13.1.1 传统程序 vs Agent

```
传统程序：
  输入 → 固定逻辑 → 输出
  特点：
    ✗ 流程固定
    ✗ 无法适应变化
    ✗ 需要人工编码所有逻辑

Agent：
  目标 → 感知 → 决策 → 行动 → 反馈 → 调整
  特点：
    ✓ 自主决策
    ✓ 环境感知
    ✓ 持续学习
    ✓ 目标导向
```

### 13.1.2 Agent的核心组件

```python
# Agent架构示意

class Agent:
    """
    Agent = LLM + 工具 + 记忆 + 规划
    """

    def __init__(self):
        # 1. 大脑（LLM）
        self.llm = LLM()

        # 2. 工具集
        self.tools = {
            'search': SearchTool(),
            'calculator': CalculatorTool(),
            'rag': RAGTool()
        }

        # 3. 记忆
        self.memory = Memory()

        # 4. 规划器
        self.planner = Planner()

    def run(self, task: str) -> str:
        """执行任务"""
        # 步骤1：理解任务
        understanding = self.understand(task)

        # 步骤2：制定计划
        plan = self.planner.plan(understanding)

        # 步骤3：执行计划
        for step in plan:
            result = self.execute_step(step)
            self.memory.update(result)

        # 步骤4：生成最终答案
        answer = self.generate_answer()
        return answer
```

---

## 13.2 ReAct模式

### 13.2.1 原理

**ReAct = Reasoning + Acting**

```
传统RAG：
  Query → [检索] → Answer

ReAct RAG：
  Query → [Thought 1] → [Action 1] → [Observation 1]
        → [Thought 2] → [Action 2] → [Observation 2]
        → ...
        → [Final Answer]

示例：

Query: "马斯克的火箭公司最近一次发射是什么时候？"

Thought 1: 需要先确定马斯克的火箭公司名称
Action 1: Search["马斯克火箭公司"]
Observation 1: SpaceX

Thought 2: 现在需要查找SpaceX最近的发射
Action 2: Search["SpaceX latest launch"]
Observation 2: Starship第三次试飞，2024年3月14日

Thought 3: 我有了足够信息回答
Action 3: Finish["SpaceX最近一次发射是2024年3月14日的Starship第三次试飞"]
```

### 13.2.2 完整实现

```python
# 文件名：react_agent.py
"""
ReAct Agent实现
"""

from typing import List, Dict, Callable, Any
import json
import re


class ReActAgent:
    """
    ReAct Agent

    结合推理(Reasoning)和行动(Acting)

    Args:
        llm_client: LLM客户端
        tools: 工具字典
        max_iterations: 最大迭代次数

    Example:
        >>> agent = ReActAgent(llm_client, tools)
        >>> result = agent.run("Python和JavaScript的优缺点对比")
    """

    def __init__(self, llm_client, tools: Dict[str, Callable],
                 max_iterations: int = 10):
        self.llm_client = llm_client
        self.tools = tools
        self.max_iterations = max_iterations

        # ReAct提示模板
        self.prompt_template = """
你是一个智能助手，可以使用工具来回答问题。

可用工具：
{tools}

回答格式：
Thought: 你的思考过程
Action: 工具名称[参数]
Observation: 工具返回的结果
... (重复Thought/Action/Observation)
Thought: 我知道了最终答案
Action: Finish[最终答案]

开始！

Question: {query}
Thought: {previous_thought}
"""

    def run(self, query: str) -> Dict:
        """
        运行Agent

        Args:
            query: 用户查询

        Returns:
            {
                'answer': str,
                'steps': List[Dict],
                'iterations': int
            }
        """
        steps = []
        thought = ""

        for iteration in range(self.max_iterations):
            # 构建提示
            prompt = self._build_prompt(query, steps, thought)

            # 调用LLM
            response = self.llm_client.generate(prompt)

            # 解析响应
            thought, action, action_input = self._parse_response(response)

            # 记录步骤
            step = {
                'iteration': iteration + 1,
                'thought': thought,
                'action': action,
                'action_input': action_input
            }
            steps.append(step)

            # 检查是否结束
            if action == "Finish":
                return {
                    'answer': action_input,
                    'steps': steps,
                    'iterations': iteration + 1
                }

            # 执行行动
            if action in self.tools:
                observation = self.tools[action](action_input)
                step['observation'] = observation
                thought = f"Observation: {observation}"
            else:
                thought = f"Error: 工具 '{action}' 不存在"
                step['error'] = True

        # 达到最大迭代次数
        return {
            'answer': "抱歉，未能在给定步骤内完成",
            'steps': steps,
            'iterations': self.max_iterations
        }

    def _build_prompt(self, query: str, steps: List[Dict],
                     previous_thought: str) -> str:
        """构建提示"""
        # 工具描述
        tools_desc = "\n".join([
            f"  - {name}: {tool.__doc__ or tool.__name__}"
            for name, tool in self.tools.items()
        ])

        # 历史步骤
        history = "\n".join([
            f"Thought: {step['thought']}\n"
            f"Action: {step['action']}[{step['action_input']}]"
            + (f"\nObservation: {step.get('observation', '')}"
                if 'observation' in step else "")
            for step in steps
        ])

        # 构建完整提示
        prompt = self.prompt_template.format(
            tools=tools_desc,
            query=query,
            previous_thought=previous_thought or "让我分析一下这个问题..."
        )

        if history:
            prompt += f"\n\n{history}"

        prompt += "\n\nThought:"

        return prompt

    def _parse_response(self, response: str) -> tuple:
        """解析LLM响应"""
        # 提取Thought
        thought_match = re.search(r"Thought: (.+?)(?:\n|$)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # 提取Action
        action_match = re.search(r"Action: (\w+)\[(.+?)\]", response, re.DOTALL)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2).strip()
        else:
            # 可能是Finish
            finish_match = re.search(r"Action: Finish\[(.+?)\]", response, re.DOTALL)
            if finish_match:
                action = "Finish"
                action_input = finish_match.group(1).strip()
            else:
                action = "Error"
                action_input = "无法解析响应"

        return thought, action, action_input


# 工具定义示例
class RAGTool:
    """RAG检索工具"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def __call__(self, query: str) -> str:
        """执行检索"""
        result = self.rag_system.query(query)
        return f"找到{len(result['sources'])}个相关文档，答案：{result['answer'][:200]}"


class SearchTool:
    """搜索工具"""

    def __call__(self, query: str) -> str:
        """执行搜索"""
        # 实际使用时调用搜索API
        return f"搜索'{query}'的结果：这里是模拟的搜索结果..."


class CalculatorTool:
    """计算器工具"""

    def __call__(self, expression: str) -> str:
        """执行计算"""
        try:
            result = eval(expression)
            return f"计算结果：{result}"
        except:
            return "计算错误"


# 使用示例
if __name__ == "__main__":
    # 模拟LLM
    class MockLLM:
        def generate(self, prompt: str) -> str:
            # 简化示例：返回固定响应
            if "第一步" in prompt or "Let me think" in prompt:
                return """Action: Search[Python特点]"""
            else:
                return """Action: Finish[Python和JavaScript各有优势...]"""

    llm = MockLLM()

    # 准备工具
    tools = {
        'Search': SearchTool(),
        'RAG': RAGTool(None),
        'Calculator': CalculatorTool()
    }

    # 创建Agent
    agent = ReActAgent(llm, tools, max_iterations=5)

    # 运行
    result = agent.run("Python和JavaScript的优缺点对比")

    print(f"\n最终答案:\n{result['answer']}")
    print(f"\n执行步骤数: {result['iterations']}")
```

---

## 13.3 LangChain Agent

### 13.3.1 快速开始

```python
# 文件名：langchain_agent.py
"""
使用LangChain构建Agent
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub


# 定义工具
def search_tool(query: str) -> str:
    """搜索工具"""
    return f"关于'{query}'的搜索结果..."

def rag_tool(query: str) -> str:
    """RAG检索工具"""
    return f"RAG检索'{query}'的结果..."

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="用于搜索最新信息"
    ),
    Tool(
        name="RAG",
        func=rag_tool,
        description="用于检索知识库中的信息"
    )
]

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 获取ReAct提示模板
prompt = hub.pull("hwchase17/react")

# 创建Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 创建Agent执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 运行
result = agent_executor.invoke({
    "input": "Python的性能优化有哪些方法？"
})

print(f"\n最终答案: {result['output']}")
```

---

## 13.4 综合项目：智能问答Agent

### 13.4.1 项目设计

**项目名称**：InteliKB Research Agent

**功能需求**：
1. 支持多轮对话
2. 能够使用多个工具
3. 自主规划检索策略
4. 提供可解释的推理过程

**技术栈**：
- LangChain
- OpenAI GPT-3.5
- 自定义RAG工具
- 内存管理

### 13.4.2 完整实现

```python
# 文件名：research_agent.py
"""
研究助手Agent
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub


class ResearchAgent:
    """
    研究助手Agent

    集成多种工具，支持复杂查询和多轮对话
    """

    def __init__(self, openai_api_key: str):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key
        )

        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 创建工具
        self.tools = self._create_tools()

        # 创建Agent
        prompt = hub.pull("hwchase17/react-chat")
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # 创建执行器
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def _create_tools(self) -> List[Tool]:
        """创建工具集"""
        tools = [
            Tool(
                name="KnowledgeBase",
                func=self._rag_retrieve,
                description="用于检索知识库中的技术文档、教程等。输入应该是具体的问题。"
            ),
            Tool(
                name="WebSearch",
                func=self._web_search,
                description="用于搜索互联网上的最新信息。输入应该是搜索关键词。"
            ),
            Tool(
                name="CodeSearch",
                func=self._code_search,
                description="用于搜索代码示例和实现。输入应该是编程语言或具体功能。"
            ),
            Tool(
                name="Calculator",
                func=self._calculator,
                description="用于执行数学计算。输入应该是数学表达式。"
            )
        ]
        return tools

    def _rag_retrieve(self, query: str) -> str:
        """RAG检索"""
        # 实际使用时连接真实RAG系统
        return f"从知识库检索'{query}'的相关内容..."

    def _web_search(self, query: str) -> str:
        """网络搜索"""
        # 实际使用时调用搜索API
        return f"搜索'{query}'的网络结果..."

    def _code_search(self, query: str) -> str:
        """代码搜索"""
        # 实际使用时搜索代码仓库
        return f"搜索'{query}'的代码示例..."

    def _calculator(self, expression: str) -> str:
        """计算器"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行查询

        Args:
            question: 用户问题

        Returns:
            {
                'answer': str,
                'intermediate_steps': List,
                'chat_history': List
            }
        """
        result = self.executor.invoke({"input": question})

        return {
            'answer': result.get('output', ''),
            'intermediate_steps': result.get('intermediate_steps', []),
            'chat_history': self.memory.chat_memory.messages
        }

    def chat(self, message: str) -> str:
        """
        对话接口（简化版）

        Args:
            message: 用户消息

        Returns:
            Agent回复
        """
        result = self.query(message)
        return result['answer']


# 使用示例
if __name__ == "__main__":
    import os

    # 创建Agent
    agent = ResearchAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # 对话示例
    questions = [
        "什么是Python的GIL？",
        "它如何影响多线程性能？",
        "有什么解决方案吗？"
    ]

    print("="*80)
    print("InteliKB研究助手Agent对话")
    print("="*80 + "\n")

    for i, question in enumerate(questions, 1):
        print(f"\n用户: {question}")

        answer = agent.chat(question)

        print(f"\n助手: {answer}\n")
        print("-"*80)
```

---

## 13.5 Agent评估

### 13.5.1 评估指标

```python
class AgentEvaluator:
    """
    Agent评估器

    评估维度：
    1. 任务完成率
    2. 工具使用准确率
    3. 推理步骤数
    4. 响应时间
    5. 答案质量
    """

    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'tool_accuracy': [],
            'avg_steps': [],
            'response_times': [],
            'answer_quality': []
        }

    def evaluate(self, agent: ResearchAgent,
                 test_queries: List[Dict]) -> Dict:
        """
        评估Agent性能

        Args:
            agent: 待评估的Agent
            test_queries: 测试查询列表
                每个元素为{'query': str, 'expected_tools': List[str], 'expected_answer': str}

        Returns:
            评估报告
        """
        import time

        for item in test_queries:
            query = item['query']
            expected_tools = item.get('expected_tools', [])

            # 执行查询
            start_time = time.time()
            result = agent.query(query)
            response_time = time.time() - start_time

            # 记录指标
            self.metrics['total_queries'] += 1

            # 检查是否成功
            if result['answer'] and "Error" not in result['answer']:
                self.metrics['successful_queries'] += 1

            # 工具使用准确率
            used_tools = [
                step[0].tool for step in result['intermediate_steps']
                if step[0].tool != "Final Answer"
            ]
            tool_accuracy = len(set(used_tools) & set(expected_tools)) / max(len(expected_tools), 1)
            self.metrics['tool_accuracy'].append(tool_accuracy)

            # 推理步骤数
            steps = len(result['intermediate_steps'])
            self.metrics['avg_steps'].append(steps)

            # 响应时间
            self.metrics['response_times'].append(response_time)

        # 计算总体指标
        return {
            'success_rate': self.metrics['successful_queries'] / self.metrics['total_queries'],
            'avg_tool_accuracy': sum(self.metrics['tool_accuracy']) / len(self.metrics['tool_accuracy']),
            'avg_steps': sum(self.metrics['avg_steps']) / len(self.metrics['avg_steps']),
            'avg_response_time': sum(self.metrics['response_times']) / len(self.metrics['response_times'])
        }
```

---

## 练习题

### 练习13.1：实现简单ReAct Agent（基础）

**题目**：从零实现一个ReAct Agent

**要求**：
1. 支持至少3个工具（Search、RAG、Calculator）
2. 实现Thought-Action-Observation循环
3. 最多5次迭代
4. 提供推理过程可视化

---

### 练习13.2：构建问答Agent（进阶）

**题目**：构建一个技术问答Agent

**要求**：
1. 集成RAG检索工具
2. 支持多轮对话
3. 能够调用代码搜索工具
4. 提供答案来源引用

---

### 练习13.3：研究助手Agent（挑战）

**题目**：构建完整的研究助手Agent

**功能需求**：
1. 文献搜索和总结
2. 代码示例检索
3. 技术对比分析
4. 自动生成报告
5. 多轮迭代优化

---

## 总结

### 本章要点

1. **Agent核心**
   - Agent = LLM + 工具 + 记忆 + 规划
   - 自主决策和执行
   - 持续学习和改进

2. **ReAct模式**
   - Thought → Action → Observation循环
   - 推理与行动结合
   - 可解释的执行过程

3. **LangChain Agent**
   - 快速构建Agent
   - 丰富的工具生态
   - 易于扩展和定制

### 学习检查清单

- [ ] 理解Agent的工作原理
- [ ] 掌握ReAct模式
- [ ] 能够实现简单Agent
- [ ] 了解LangChain Agent框架
- [ ] 完成研究助手Agent项目

### 下一步学习

- **下一章**：[第14章：高级Agent模式](./14-高级Agent模式.md)
- **相关资源**：
  - LangChain Agents: https://python.langchain.com/docs/modules/agents/
  - ReAct Paper: https://arxiv.org/abs/2210.03629

---

**恭喜完成第13章！** 🎉

**继续学习高级Agent模式！** → [第14章](./14-高级Agent模式.md)
