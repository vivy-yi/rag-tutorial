# 第14章：高级Agent模式

> 超越ReAct：掌握Plan-and-Execute、自主规划和多Agent协作，构建能够处理复杂任务的智能系统！

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解Plan-and-Execute模式
- [ ] 实现自主规划Agent
- [ ] 掌握多Agent协作机制
- [ ] 理解深度推理原理 (Deep Research)
- [ ] 实现Deep Research风格Agent
- [ ] 构建研究助手Agent
- [ ] 完成复杂任务Agent项目

**预计学习时间**：8小时
**难度等级**：⭐⭐⭐⭐⭐

---

## 前置知识

- [ ] 完成第13章：Agentic RAG基础
- [ ] 熟悉ReAct模式
- [ ] 理解LangChain Agent框架
- [ ] 有实际Agent开发经验

---

## 14.1 Plan-and-Execute模式

### 14.1.1 原理

**ReAct vs Plan-and-Execute**

```
ReAct模式（第13章）：
  Thought → Action → Observation → Thought → Action → Observation → ...
  特点：边思考边执行，逐步探索

Plan-and-Execute模式：
  任务 → 制定计划 → 执行所有步骤 → 总结
  特点：先规划后执行，系统性强

示例：复杂研究任务

ReAct方式：
  Q: "对比Python和JavaScript在Web开发中的差异"
  T1: 需要了解Python在Web开发中的应用
  A1: Search["Python web development"]
  O1: ...
  T2: 需要了解JavaScript的特点
  A2: Search["JavaScript features"]
  O2: ...
  ...（可能10+轮迭代）

Plan-and-Execute方式：
  Q: "对比Python和JavaScript在Web开发中的差异"

  计划阶段：
    Step 1: 了解Python的Web框架和应用场景
    Step 2: 了解JavaScript的特点和应用场景
    Step 3: 对比两者的性能、学习曲线、生态系统
    Step 4: 总结各自的优势和劣势

  执行阶段：
    执行Step 1 → 收集信息
    执行Step 2 → 收集信息
    执行Step 3 → 对比分析
    执行Step 4 → 总结报告

  优势：清晰、可控、可追溯
```

### 14.1.2 完整实现

```python
# 文件名：plan_execute_agent.py
"""
Plan-and-Execute Agent实现
"""

from typing import List, Dict, Any
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class PlanExecuteAgent:
    """
    Plan-and-Execute Agent

    工作流程：
    1. 接收任务
    2. 制定详细计划
    3. 执行每个步骤
    4. 汇总结果

    Args:
        llm: LLM实例
        tools: 可用工具字典

    Example:
        >>> agent = PlanExecuteAgent(llm, tools)
        >>> result = agent.run("分析2024年AI领域的发展趋势")
    """

    def __init__(self, llm, tools: Dict[str, callable]):
        self.llm = llm
        self.tools = tools

        # 规划提示模板
        self.planning_prompt = PromptTemplate(
            input_variables=["task"],
            template="""你是任务规划专家。请为以下任务制定详细的执行计划。

任务：{task}

请制定一个详细的计划，包括：
1. 任务分析：这个任务的目标是什么？需要什么信息？
2. 拆解步骤：将任务拆分为3-7个具体步骤
3. 每个步骤说明：明确每个步骤要做什么、使用什么工具

输出格式（JSON）：
{{
    "analysis": "任务分析",
    "steps": [
        {{"step": 1, "action": "ToolName", "description": "...", "input": "..."}},
        {{"step": 2, "action": "ToolName", "description": "...", "input": "..."}}
    ]
}}

请只返回JSON，不要其他内容。"""
        )

        # 执行提示模板
        self.execution_prompt = PromptTemplate(
            input_variables=["step", "task", "previous_results"],
            template="""执行任务的第{step}步。

任务：{task}

这一步的说明：{step_description}

之前步骤的结果：
{previous_results}

请执行这一步，并返回结果。

输出格式：
Result: [具体结果]

请只返回Result和内容。"""
        )

    def plan(self, task: str) -> Dict[str, Any]:
        """制定计划"""
        print("\n=== 规划阶段 ===")
        print(f"任务: {task}\n")

        # 生成计划
        prompt = self.planning_prompt.format(task=task)
        response = self.llm.predict(prompt)

        try:
            plan = json.loads(response)
            print("✓ 计划已制定")
            print(f"  分析: {plan['analysis']}")
            print(f"  步骤数: {len(plan['steps'])}")

            for step in plan['steps']:
                print(f"  Step {step['step']}: {step['action']} - {step['description']}")

            return plan
        except:
            print("⚠ 计划解析失败，使用默认计划")
            return self._default_plan(task)

    def _default_plan(self, task: str) -> Dict:
        """默认计划（后备方案）"""
        return {
            "analysis": "分析该任务",
            "steps": [
                {"step": 1, "action": "Search", "description": "搜索相关信息", "input": task},
                {"step": 2, "action": "RAG", "description": "检索知识库", "input": task},
                {"step": 3, "action": "Summarize", "description": "总结结果", "input": ""}
            ]
        }

    def execute(self, plan: Dict, task: str) -> Dict:
        """执行计划"""
        print("\n=== 执行阶段 ===")

        results = []
        previous_summary = ""

        for step_info in plan['steps']:
            step_num = step_info['step']
            action = step_info['action']
            description = step_info['description']
            step_input = step_info.get('input', task)

            print(f"\n执行Step {step_num}: {description}")

            # 执行步骤
            try:
                if action in self.tools:
                    result = self.tools[action](step_input)
                elif action == "Summarize":
                    result = self._summarize(task, results)
                else:
                    result = f"未知工具: {action}"

                step_result = {
                    "step": step_num,
                    "action": action,
                    "description": description,
                    "result": result
                }
                results.append(step_result)

                print(f"✓ Step {step_num}完成")

            except Exception as e:
                print(f"✗ Step {step_num}失败: {e}")
                results.append({
                    "step": step_num,
                    "action": action,
                    "error": str(e)
                })

        # 汇总
        print("\n=== 结果汇总 ===")
        final_answer = self._summarize(task, results)
        print(f"\n最终答案:\n{final_answer}")

        return {
            "plan": plan,
            "results": results,
            "final_answer": final_answer
        }

    def _summarize(self, task: str, results: List[Dict]) -> str:
        """汇总结果"""
        results_text = "\n".join([
            f"Step {r['step']} ({r['action']}): {r.get('result', r.get('error', 'N/A'))}"
            for r in results
        ])

        summary_prompt = f"""基于以下执行步骤的结果，提供最终答案。

任务：{task}

执行步骤：
{results_text}

请提供一个全面、准确的最终答案。"""

        return self.llm.predict(summary_prompt)

    def run(self, task: str) -> Dict:
        """完整运行Plan-and-Execute流程"""
        # 步骤1：规划
        plan = self.plan(task)

        # 步骤2：执行
        result = self.execute(plan, task)

        return result


# 使用示例
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    import os

    # 模拟LLM
    class MockLLM:
        def predict(self, prompt):
            if "计划" in prompt or "plan" in prompt:
                return json.dumps({
                    "analysis": "分析任务需求",
                    "steps": [
                        {"step": 1, "action": "Search", "description": "搜索Python特点", "input": "Python特性"},
                        {"step": 2, "action": "Search", "description": "搜索JavaScript特点", "input": "JavaScript特性"},
                        {"step": 3, "action": "Summarize", "description": "总结对比", "input": ""}
                    ]
                })
            return "模拟响应"

    # 创建工具
    tools = {
        "Search": lambda q: f"关于'{q}'的搜索结果...",
        "RAG": lambda q: f"RAG检索'{q}'的结果..."
    }

    # 创建Agent
    llm = MockLLM()
    agent = PlanExecuteAgent(llm, tools)

    # 运行
    task = "对比Python和JavaScript在Web开发中的差异"
    result = agent.run(task)
```

---

## 14.2 多Agent协作

### 14.2.1 协作模式

```
多Agent协作模式：

1. 层级式（Hierarchical）
   Manager Agent
     ├─ Research Agent
     ├─ Writing Agent
     └─ Review Agent

2. 平等式（Egalitarian）
   Agent A ──────┐
                ├─ 讨论与投票
   Agent B ──────┤
                │
   Agent C ──────┘

3. 顺序式（Sequential）
   Agent A → Agent B → Agent C
   (流水线处理)

4. 竞争式（Competitive）
   Agent A ─┐
            ├─ 提供方案 → 评估 → 选最优
   Agent B ─┘
```

### 14.2.2 实现：层级式多Agent系统

```python
# 文件名：multi_agent_system.py
"""
多Agent协作系统
"""

from typing import List, Dict
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Agent基类"""

    @abstractmethod
    def execute(self, task: str) -> Dict:
        """执行任务"""
        pass

    @abstractmethod
    def get_role(self) -> str:
        """获取Agent角色"""
        pass


class ResearchAgent(BaseAgent):
    """研究Agent"""

    def __init__(self, name: str, tools: Dict):
        self.name = name
        self.tools = tools

    def execute(self, task: str) -> Dict:
        """执行研究任务"""
        print(f"[{self.name}] 开始研究: {task}")

        # 搜索和检索
        search_result = self.tools['Search'](task)
        rag_result = self.tools['RAG'](task)

        result = f"研究结果: {search_result}\n{rag_result}"
        return {"agent": self.name, "result": result}

    def get_role(self) -> str:
        return "研究"


class WritingAgent(BaseAgent):
    """写作Agent"""

    def __init__(self, name: str):
        self.name = name

    def execute(self, task: str) -> Dict:
        """执行写作任务"""
        print(f"[{self.name}] 开始写作: {task}")

        # 基于研究结果生成内容
        prompt = f"基于以下信息，撰写详细报告：{task}"
        content = f"为'{task}'撰写的详细报告..."

        return {"agent": self.name, "content": content}

    def get_role(self) -> str:
        return "写作"


class ReviewAgent(BaseAgent):
    """审核Agent"""

    def __init__(self, name: str):
        self.name = name

    def execute(self, task: Dict) -> Dict:
        """审核内容"""
        print(f"[{self.name}] 开始审核")

        # 评估质量
        review = self._review_content(task)

        return {"agent": self.name, "review": review}

    def _review_content(self, content: Dict) -> str:
        """审核内容质量"""
        return f"审核意见：内容结构清晰，但需要补充细节..." + f"\n原始内容: {str(content)[:100]}..."

    def get_role(self) -> str:
        return "审核"


class ManagerAgent:
    """管理Agent（协调者）"""

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def coordinate(self, task: str) -> Dict:
        """协调多Agent完成任务"""
        print("\n=== 多Agent协作开始 ===")
        print(f"任务: {task}\n")

        # 分配任务
        current_result = {"task": task}

        for agent in self.agents:
            print(f"\n--- 分配给{agent.get_role()}Agent ---")

            # 执行任务
            if isinstance(agent, ReviewAgent):
                # ReviewAgent需要前面的结果
                result = agent.execute(current_result)
            else:
                result = agent.execute(task)

            # 更新结果
            current_result.update(result)

        print("\n=== 多Agent协作完成 ===")
        return current_result


# 使用示例
if __name__ == "__main__":
    # 创建工具
    tools = {
        "Search": lambda q: f"搜索'{q}'的结果...",
        "RAG": lambda q: f"检索'{q}'的结果..."
    }

    # 创建Agents
    researcher = ResearchAgent("研究助手", tools)
    writer = WritingAgent("写作助手")
    reviewer = ReviewAgent("审核助手")

    # 创建管理Agent
    manager = ManagerAgent([researcher, writer, reviewer])

    # 执行任务
    task = "撰写一篇关于RAG技术发展的报告"
    final_result = manager.coordinate(task)

    print(f"\n最终结果:\n{final_result}")
```

---

## 14.3 自主规划Agent

### 14.3.1 自我反思机制

```python
# 文件名：self_reflection_agent.py
"""
带自我反思的Agent
"""

class SelfReflectionAgent:
    """
    自我反思Agent

    特点：
    1. 执行任务
    2. 评估结果质量
    3. 反思不足
    4. 改进并重试
    """

    def __init__(self, llm, tools, max_reflections=2):
        self.llm = llm
        self.tools = tools
        self.max_reflections = max_reflections

    def run_with_reflection(self, task: str) -> Dict:
        """带反思的执行"""
        for attempt in range(self.max_reflections + 1):
            print(f"\n=== 尝试 {attempt + 1} ===")

            # 执行任务
            result = self._execute(task)

            # 第一次尝试不需要反思
            if attempt == 0:
                continue

            # 自我反思
            reflection = self._reflect(task, result)

            print(f"反思: {reflection}")

            # 决定是否需要改进
            if self._is_satisfied(reflection):
                print("✓ 结果满意，返回答案")
                break
            else:
                print("✗ 结果不满意，改进重试...")
                task = self._improve_task(task, reflection)

        return {
            "result": result,
            "attempts": attempt + 1,
            "reflections": reflection
        }

    def _reflect(self, task: str, result: str) -> str:
        """反思结果"""
        prompt = f"""评估以下任务结果的质量：

任务：{task}
结果：{result}

请评估：
1. 结果是否完整？
2. 是否有遗漏的信息？
3. 是否有错误？

提供改进建议。"""

        return self.llm.predict(prompt)

    def _is_satisfied(self, reflection: str) -> bool:
        """判断是否满意"""
        # 简化版：检查是否包含负面词
        negative_words = ["不足", "缺少", "错误", "应该", "建议"]
        return not any(word in reflection for word in negative_words)

    def _improve_task(self, task: str, reflection: str) -> str:
        """改进任务"""
        # 基于反思改进任务
        return f"{task}（注意：{reflection[:50]}...）"
```

---

## 14.4 深度推理Agent (Deep Research)

### 14.4.1 为什么需要深度推理？

**当前Agent的局限性**:

```
传统Agent推理:
┌─────────────────────────────────┐
│ 1. 接收任务                     │
│ 2. 快速执行                     │
│ 3. 返回答案                     │
└─────────────────────────────────┘
特点：快速但不够深入

自我反思Agent (14.3节):
┌─────────────────────────────────┐
│ 1. 执行任务                     │
│ 2. 简单评估                     │
│ 3. 改进重试 (最多2-3次)         │
└─────────────────────────────────┘
特点：有反思但深度有限
```

**Deep Research (o1风格)的优势**:

```
Deep Research Agent:
┌─────────────────────────────────┐
│ 1. 初始推理                     │
│ 2. 自我验证 (检查每个推理步骤)  │
│ 3. 识别问题/缺失信息            │
│ 4. 补充检索                     │
│ 5. 重新推理 (整合新信息)        │
│ 6. 多轮验证 (可能10+轮)         │
│ 7. 生成最终答案 + 推理链        │
└─────────────────────────────────┘
特点：深度思考、严格验证、可解释
```

**对比分析**:

| 特性 | 传统Agent | 自我反思Agent | Deep Research Agent |
|------|----------|-------------|-------------------|
| 推理轮次 | 1轮 | 2-3轮 | 10+轮 |
| 验证深度 | 无 | 结果层面 | 步骤层面 |
| 推理链输出 | 无 | 无 | 完整思维链 |
| 自我质疑 | 无 | 简单 | 深度多角度 |
| 补充检索 | 无 | 触发式 | 主动式 |
| 适用场景 | 简单任务 | 中等任务 | 复杂推理 |

---

### 14.4.2 Deep Research核心原理

**关键组件**:

1. **推理链管理器** (Reasoning Chain Manager)
   - 记录每一步推理
   - 维护推理历史
   - 追踪推理依赖

2. **自我验证器** (Self-Verifier)
   - 验证每个推理步骤
   - 检查逻辑一致性
   - 评估证据充分性

3. **缺失信息识别器** (Gap Detector)
   - 主动发现知识缺口
   - 生成补充查询
   - 触发补充检索

4. **推理链优化器** (Chain Refiner)
   - 合并冗余步骤
   - 优化推理路径
   - 提炼关键洞察

**工作流程**:

```
输入任务
   ↓
[轮次1: 初始推理]
   ├─ 推理步骤1
   ├─ 推理步骤2
   └─ 推理步骤3
   ↓
[自我验证]
   ├─ 检查逻辑一致性
   ├─ 评估证据质量
   └─ 识别缺失信息
   ↓
是否需要补充?
   ├─ 是 → [补充检索] → [重新推理]
   └─ 否 → 继续验证
   ↓
[轮次2-10: 深化推理]
   ├─ 整合新信息
   ├─ 优化推理链
   └─ 多角度验证
   ↓
[最终检查]
   ├─ 推理链完整?
   ├─ 逻辑严密?
   └─ 证据充分?
   ↓
输出: 答案 + 完整推理链
```

---

### 14.4.3 完整实现

```python
# 文件名：deep_research_agent.py
"""
Deep Research Agent - 深度推理Agent实现

类似于OpenAI o1的推理模式，通过多轮自我验证和优化
生成高质量的深度推理结果。
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReasoningStep:
    """
    推理步骤数据类

    Attributes:
        step_id: 步骤编号
        content: 推理内容
        evidence: 支持证据
        confidence: 置信度 (0-1)
        questions: 产生的疑问
        dependencies: 依赖的前置步骤
    """
    step_id: int
    content: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    questions: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "questions": self.questions,
            "dependencies": self.dependencies,
            "timestamp": self.timestamp
        }


@dataclass
class VerificationResult:
    """
    验证结果数据类

    Attributes:
        is_valid: 是否通过验证
        confidence: 整体置信度
        issues: 发现的问题
        missing_info: 缺失的信息
        improvements: 改进建议
    """
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


class DeepResearchAgent:
    """
    Deep Research Agent

    特点:
    1. 长推理链 (10+轮)
    2. 每步自我验证
    3. 主动补充检索
    4. 推理链可视化
    5. 多角度自我质疑

    Args:
        llm: LLM实例
        retriever: 检索器实例
        max_rounds: 最大推理轮数 (默认10)
        confidence_threshold: 置信度阈值 (默认0.85)

    Example:
        >>> agent = DeepResearchAgent(llm, retriever)
        >>> result = agent.reason("证明费马小定理")
        >>> print(result['answer'])
        >>> print(result['reasoning_chain'])
    """

    def __init__(self,
                 llm,
                 retriever=None,
                 max_rounds: int = 10,
                 confidence_threshold: float = 0.85):

        self.llm = llm
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold

        # 推理链存储
        self.reasoning_chain: List[ReasoningStep] = []

        # 统计信息
        self.stats = {
            "total_rounds": 0,
            "total_retrievals": 0,
            "total_verifications": 0,
            "confidence_history": []
        }

    def reason(self, task: str) -> Dict[str, Any]:
        """
        深度推理主方法

        Args:
            task: 推理任务

        Returns:
            {
                'answer': str,              # 最终答案
                'reasoning_chain': List,    # 完整推理链
                'confidence': float,        # 最终置信度
                'stats': Dict               # 统计信息
            }
        """
        print("="*80)
        print("Deep Research Agent - 深度推理模式")
        print("="*80)
        print(f"\n任务: {task}")
        print(f"最大轮次: {self.max_rounds}")
        print(f"置信度阈值: {self.confidence_threshold}\n")

        # 初始化
        self.reasoning_chain = []
        context = []
        current_confidence = 0.0

        # 推理循环
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'='*80}")
            print(f"推理轮次 {round_num}/{self.max_rounds}")
            print(f"{'='*80}")

            # 步骤1: 生成推理步骤
            step = self._generate_reasoning_step(task, context, round_num)
            self.reasoning_chain.append(step)
            print(f"\n推理步骤 {step.step_id}:")
            print(f"  内容: {step.content[:100]}...")
            print(f"  置信度: {step.confidence:.2%}")

            # 步骤2: 自我验证
            print(f"\n>>> 验证推理步骤...")
            verification = self._verify_reasoning_step(step, context)

            self.stats['total_verifications'] += 1
            print(f"  验证结果: {'✓ 通过' if verification.is_valid else '✗ 未通过'}")
            print(f"  整体置信度: {verification.confidence:.2%}")

            if verification.issues:
                print(f"  发现问题: {len(verification.issues)}个")
                for issue in verification.issues[:3]:
                    print(f"    - {issue}")

            # 步骤3: 检查是否需要补充信息
            if verification.missing_info:
                print(f"\n>>> 检测到缺失信息: {len(verification.missing_info)}项")
                for info in verification.missing_info[:3]:
                    print(f"    - {info}")

                # 补充检索
                if self.retriever:
                    print(f"\n>>> 执行补充检索...")
                    new_context = self._retrieve_missing_info(
                        verification.missing_info
                    )
                    context.extend(new_context)
                    self.stats['total_retrievals'] += 1
                    print(f"  新增上下文: {len(new_context)}条")

            # 更新置信度
            current_confidence = verification.confidence
            self.stats['confidence_history'].append(current_confidence)

            # 步骤4: 检查是否可以结束
            if self._should_stop(verification, round_num):
                print(f"\n✓ 推理完成!")
                print(f"  最终置信度: {current_confidence:.2%}")
                print(f"  总轮次: {round_num}")
                break

        # 生成最终答案
        print(f"\n{'='*80}")
        print("生成最终答案...")
        print(f"{'='*80}\n")

        final_answer = self._generate_final_answer(task, self.reasoning_chain)

        self.stats['total_rounds'] = round_num

        return {
            'answer': final_answer,
            'reasoning_chain': [step.to_dict() for step in self.reasoning_chain],
            'confidence': current_confidence,
            'stats': self.stats
        }

    def _generate_reasoning_step(self,
                                  task: str,
                                  context: List[str],
                                  round_num: int) -> ReasoningStep:
        """
        生成推理步骤

        让LLM逐步推理，产生思考过程
        """
        # 构建推理历史
        chain_summary = self._summarize_reasoning_chain()

        prompt = f"""你是深度思考的AI助手。请对以下任务进行深度推理。

任务: {task}

之前的推理:
{chain_summary}

已有信息:
{chr(10).join(f"- {c}" for c in context[-5:]) if context else "暂无"}

请进行下一步推理。要求:
1. 逐步思考，不要跳跃
2. 明确你的推理依据
3. 指出你的置信度
4. 列出产生的疑问
5. 标注依赖的前置步骤

输出格式(JSON):
{{
    "content": "这一步的推理内容...",
    "evidence": ["证据1", "证据2"],
    "confidence": 0.7,
    "questions": ["疑问1", "疑问2"],
    "dependencies": [1, 2]
}}
"""

        response = self.llm.predict(prompt)

        try:
            data = json.loads(response)
            return ReasoningStep(
                step_id=len(self.reasoning_chain) + 1,
                content=data.get('content', ''),
                evidence=data.get('evidence', []),
                confidence=float(data.get('confidence', 0.5)),
                questions=data.get('questions', []),
                dependencies=data.get('dependencies', [])
            )
        except:
            # 解析失败，返回默认步骤
            return ReasoningStep(
                step_id=len(self.reasoning_chain) + 1,
                content=response[:500],
                confidence=0.5
            )

    def _verify_reasoning_step(self,
                                step: ReasoningStep,
                                context: List[str]) -> VerificationResult:
        """
        验证推理步骤

        从多个角度验证推理的合理性
        """
        prompt = f"""你是严格的推理验证专家。请验证以下推理步骤。

推理步骤: {step.step_id}
内容: {step.content}
证据: {step.evidence}
置信度: {step.confidence}

已有信息:
{chr(10).join(f"- {c}" for c in context[-3:]) if context else "暂无"}

请从以下角度验证:
1. 逻辑一致性: 推理是否严谨?是否有逻辑漏洞?
2. 证据充分性: 证据是否足够支撑结论?
3. 置信度评估: 声称的置信度是否合理?
4. 缺失信息: 是否缺少关键信息?

输出格式(JSON):
{{
    "is_valid": true/false,
    "confidence": 0.75,
    "issues": ["问题1", "问题2"],
    "missing_info": ["缺失信息1", "缺失信息2"],
    "improvements": ["改进建议1"]
}}
"""

        response = self.llm.predict(prompt)

        try:
            data = json.loads(response)
            return VerificationResult(
                is_valid=data.get('is_valid', False),
                confidence=float(data.get('confidence', step.confidence)),
                issues=data.get('issues', []),
                missing_info=data.get('missing_info', []),
                improvements=data.get('improvements', [])
            )
        except:
            # 解析失败，返回默认验证结果
            return VerificationResult(
                is_valid=step.confidence > 0.5,
                confidence=step.confidence,
                issues=[],
                missing_info=[],
                improvements=[]
            )

    def _retrieve_missing_info(self, missing_info: List[str]) -> List[str]:
        """
        补充检索缺失信息
        """
        retrieved = []

        for info in missing_info[:3]:  # 最多检索3项
            try:
                # 使用检索器查询
                docs = self.retriever.retrieve(info, top_k=2)
                for doc in docs[:2]:
                    retrieved.append(doc.get('text', '')[:200])
            except:
                pass

        return retrieved

    def _should_stop(self,
                     verification: VerificationResult,
                     round_num: int) -> bool:
        """
        判断是否应该停止推理
        """
        # 条件1: 置信度足够高
        if verification.confidence >= self.confidence_threshold:
            return True

        # 条件2: 验证通过且无缺失信息
        if verification.is_valid and not verification.missing_info:
            return True

        # 条件3: 达到最小轮次且问题不多
        if round_num >= 3 and len(verification.issues) == 0:
            return True

        return False

    def _summarize_reasoning_chain(self) -> str:
        """
        总结推理链
        """
        if not self.reasoning_chain:
            return "开始推理..."

        summary = []
        for step in self.reasoning_chain[-3:]:  # 只显示最近3步
            summary.append(f"步骤{step.step_id}: {step.content[:80]}...")

        return "\n".join(summary) if summary else "开始推理..."

    def _generate_final_answer(self,
                                task: str,
                                reasoning_chain: List[ReasoningStep]) -> str:
        """
        基于推理链生成最终答案
        """
        # 构建推理链摘要
        chain_text = "\n\n".join([
            f"步骤{s.step_id}: {s.content}"
            for s in reasoning_chain
        ])

        prompt = f"""基于以下深度推理过程，生成最终答案。

任务: {task}

推理过程:
{chain_text}

请生成:
1. 清晰直接的答案
2. 关键推理步骤总结
3. 最终结论

输出格式:
## 答案
[直接回答问题]

## 推理要点
- 要点1
- 要点2

## 结论
[最终结论]
"""

        return self.llm.predict(prompt)

    def visualize_reasoning_chain(self) -> str:
        """
        可视化推理链

        返回Markdown格式的推理链展示
        """
        if not self.reasoning_chain:
            return "暂无推理链"

        lines = ["# 深度推理链\n"]

        for step in self.reasoning_chain:
            lines.append(f"## 步骤 {step.step_id}")
            lines.append(f"**内容**: {step.content}")
            lines.append(f"**置信度**: {step.confidence:.2%}")

            if step.evidence:
                lines.append(f"**证据**:")
                for ev in step.evidence:
                    lines.append(f"  - {ev}")

            if step.questions:
                lines.append(f"**疑问**:")
                for q in step.questions:
                    lines.append(f"  - {q}")

            if step.dependencies:
                lines.append(f"**依赖**: 步骤 {', '.join(map(str, step.dependencies))}")

            lines.append("")

        return "\n".join(lines)


# 使用示例
if __name__ == "__main__":
    # 模拟LLM
    class MockLLM:
        def predict(self, prompt):
            # 根据提示词返回模拟响应
            if "下一步推理" in prompt:
                return json.dumps({
                    "content": "基于已有信息，我认为这个问题的关键在于...",
                    "evidence": ["根据经验法则...", "从数学角度..."],
                    "confidence": 0.75,
                    "questions": ["是否还有其他可能性?"],
                    "dependencies": [len(prompt.split("推理步骤")) > 0]
                })
            elif "验证" in prompt:
                return json.dumps({
                    "is_valid": True,
                    "confidence": 0.8,
                    "issues": [],
                    "missing_info": ["需要更多数据支持"],
                    "improvements": ["可以进一步验证"]
                })
            return "模拟响应"

    # 模拟检索器
    class MockRetriever:
        def retrieve(self, query, top_k=3):
            return [
                {'text': f'关于"{query}"的相关信息...'}
            ]

    # 创建Deep Research Agent
    llm = MockLLM()
    retriever = MockRetriever()

    agent = DeepResearchAgent(
        llm=llm,
        retriever=retriever,
        max_rounds=5,
        confidence_threshold=0.85
    )

    # 执行深度推理
    task = "解释量子纠缠的原理，并说明其在量子计算中的应用"
    result = agent.reason(task)

    # 输出结果
    print("\n" + "="*80)
    print("最终答案:")
    print("="*80)
    print(result['answer'])

    print("\n" + "="*80)
    print("推理链可视化:")
    print("="*80)
    print(agent.visualize_reasoning_chain())

    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    print(json.dumps(result['stats'], indent=2, ensure_ascii=False))
```

---

### 14.4.4 实战应用案例

**案例1: 复杂数学问题推理**

```python
# 任务: 证明一个数学定理
task = "证明欧拉公式 e^(iπ) + 1 = 0"

agent = DeepResearchAgent(llm, retriever, max_rounds=15)
result = agent.reason(task)

# 输出包含:
# - 泰勒级数展开
# - 欧拉公式推导
# - 几何意义解释
# - 验证每个步骤
```

**案例2: 技术方案对比分析**

```python
# 任务: 深度对比多种技术方案
task = """深度分析以下RAG优化方案的优劣势:
1. 混合检索 (Vector + BM25)
2. 迭代检索
3. 重排序
4. GraphRAG

请从性能、成本、效果三个维度分析。"""

result = agent.reason(task)

# 推理链会包含:
# - 逐一分析每个方案
# - 横向对比
# - 补充检索最新研究
# - 生成推荐建议
```

**案例3: 代码调试与优化**

```python
# 任务: 深度分析代码问题
task = f"""
分析以下代码的性能问题:

```python
def process_data(data):
    results = []
    for item in data:
        for key, value in item.items():
            results.append(transform(key, value))
    return results
```

请识别问题并提供优化方案。
"""

result = agent.reason(task)

# 推理链会包含:
# - 时间复杂度分析
# - 空间复杂度分析
# - 瓶颈识别
# - 优化建议
# - 验证改进效果
```

---

### 14.4.5 Deep Research vs 其他Agent模式

**模式对比**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Agent模式对比                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式              推理深度  验证强度  适用场景      成本   │
│  ─────────────────────────────────────────────────────    │
│  ReAct            ⭐⭐    ⭐       快速问答      低     │
│  Plan-Execute     ⭐⭐⭐   ⭐⭐      任务执行      中     │
│  Self-Reflection  ⭐⭐⭐   ⭐⭐⭐    质量优化      中     │
│  Deep Research    ⭐⭐⭐⭐⭐ ⭐⭐⭐⭐⭐  复杂推理      高     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**选择建议**:

- **简单问答** → ReAct (快速响应)
- **任务执行** → Plan-and-Execute (清晰可控)
- **质量优化** → Self-Reflection (提升准确率)
- **复杂推理** → Deep Research (深度分析)

---

## 14.5 完整项目：研究助手Agent

### 14.4.1 项目设计

**项目名称**：AI Research Agent

**功能需求**：
1. 文献搜索和总结
2. 技术对比分析
3. 代码示例检索
4. 自动生成报告
5. 多轮迭代优化

**Agent分工**：
- **ManagerAgent**：整体协调
- **SearchAgent**：网络搜索
- **RAGAgent**：知识库检索
- **AnalysisAgent**：数据分析
- **WritingAgent**：报告生成

### 14.4.2 完整实现

```python
# 文件名：research_agent_complete.py
"""
完整的研究助手Agent系统
"""

from typing import List, Dict, Any
import json


class AIResearchAgent:
    """
    AI研究助手系统

    集成多个专门Agent，自动完成研究任务
    """

    def __init__(self, openai_api_key: str):
        from langchain_openai import ChatOpenAI
        from langchain.agents import initialize_agent, Tool, AgentExecutor

        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)

        # 定义工具
        self.tools = self._create_tools()

        # 创建Agent
        self.agent = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """创建工具集"""
        tools = [
            Tool(
                name="LiteratureSearch",
                func=self._literature_search,
                description="搜索学术文献和最新研究。输入：搜索关键词。"
            ),
            Tool(
                name="CodeSearch",
                func=self._code_search,
                description="搜索代码示例和实现。输入：编程语言或功能描述。"
            ),
            Tool(
                name="KnowledgeQuery",
                func=self._knowledge_query,
                description="查询内部知识库。输入：具体问题。"
            ),
            Tool(
                name="DataAnalysis",
                func=self._data_analysis,
                description="分析数据并生成报告。输入：数据描述或指标。"
            )
        ]
        return tools

    def _literature_search(self, query: str) -> str:
        """文献搜索"""
        # 实际实现调用arXiv、Google Scholar等API
        return f"文献搜索结果：找到关于'{query}'的10篇相关论文..."

    def _code_search(self, query: str) -> str:
        """代码搜索"""
        # 实际实现搜索GitHub、Stack Overflow等
        return f"代码搜索结果：找到'{query}'的5个代码示例..."

    def _knowledge_query(self, query: str) -> str:
        """知识库查询"""
        # 实际实现查询向量数据库
        return f"知识库查询结果：{query}的相关信息..."

    def _data_analysis(self, data_desc: str) -> str:
        """数据分析"""
        # 实际实现分析数据
        return f"数据分析结果：基于{data_desc}的分析..."

    def _create_agent(self):
        """创建Agent"""
        from langchain.agents import initialize_agent, AgentExecutor
        from langchain import hub

        prompt = hub.pull("hwchase17/react-chat")

        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="chat-zero-shot-react-description",
            verbose=True
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True
        )

    def research(self, topic: str, depth: str = "medium") -> Dict:
        """
        执行研究任务

        Args:
            topic: 研究主题
            depth: 研究深度（shallow/medium/deep）
        """
        print("="*80)
        print(f"AI研究助手")
        print("="*80)
        print(f"\n研究主题: {topic}")
        print(f"研究深度: {depth}\n")

        # 构建研究计划
        research_plan = self._build_research_plan(topic, depth)

        print("研究计划:")
        for i, step in enumerate(research_plan['steps'], 1):
            print(f"  {i}. {step}")

        # 执行研究
        results = []
        for step in research_plan['steps']:
            print(f"\n>>> 执行: {step}")

            result = self.agent.invoke({"input": f"{topic} - {step}"})
            results.append({
                "step": step,
                "result": result['output']
            })

        # 生成报告
        report = self._generate_report(topic, results)

        return {
            "topic": topic,
            "plan": research_plan,
            "results": results,
            "report": report
        }

    def _build_research_plan(self, topic: str, depth: str) -> Dict:
        """构建研究计划"""
        if depth == "shallow":
            steps = [
                "搜索基础定义和概念",
                "查找简单示例",
                "总结关键点"
            ]
        elif depth == "medium":
            steps = [
                "搜索理论基础和技术细节",
                "查找实际应用案例",
                "分析优缺点",
                "总结结论"
            ]
        else:  # deep
            steps = [
                "搜索最新研究进展",
                "查找学术论文",
                "搜索代码实现",
                "分析性能数据",
                "对比不同方案",
                "总结发展趋势",
                "提供实践建议"
            ]

        return {"depth": depth, "steps": steps}

    def _generate_report(self, topic: str, results: List[Dict]) -> str:
        """生成研究报告"""
        # 汇总所有结果
        all_content = "\n\n".join([
            f"## {r['step']}\n\n{r['result']}"
            for r in results
        ])

        # 使用LLM生成结构化报告
        prompt = f"""基于以下研究结果，生成一份结构化的研究报告。

主题：{topic}

研究内容：
{all_content}

请生成包含以下部分的报告：
1. 执行摘要
2. 背景介绍
3. 核心内容
4. 技术分析
5. 结论与建议

报告格式：Markdown
"""

        return self.llm.predict(prompt)


# 使用示例
if __name__ == "__main__":
    import os

    # 创建研究助手
    agent = AIResearchAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # 执行研究任务
    topic = "2024年大语言模型在RAG系统中的应用进展"

    result = agent.research(topic, depth="medium")

    # 保存报告
    with open(f"report_{topic.replace(' ', '_')[:20]}.md", "w") as f:
        f.write(result['report'])

    print(f"\n报告已生成并保存！")
```

---

## 练习题

### 练习14.1：实现Plan-and-Execute Agent（进阶）

**题目**：构建一个Plan-and-Execute Agent

**要求**：
1. 实现规划阶段（生成详细步骤）
2. 实现执行阶段（逐步执行）
3. 支持动态调整计划
4. 提供完整的执行轨迹

---

### 练习14.2：构建多Agent系统（挑战）

**题目**：实现层级式多Agent协作系统

**功能需求**：
1. Manager Agent（协调者）
2. 至少3个Worker Agents
3. 明确的分工和通信机制
4. 处理Agent失败的情况

---

### 练习14.3：自主研究Agent（综合项目）

**题目**：构建完整的研究助手Agent

**功能需求**：
1. 文献搜索和总结
2. 技术对比分析
3. 自动生成报告
4. 自我反思和改进
5. 支持多轮对话

---

## 总结

### 本章要点

1. **Plan-and-Execute**
   - 先规划后执行
   - 系统性强，可追溯
   - 适合复杂任务

2. **多Agent协作**
   - 层级式、平等式、顺序式
   - 各司其职，协同工作
   - 提升整体效率

3. **自我反思**
   - 评估结果质量
   - 持续改进优化
   - 提升最终答案质量

### 学习检查清单

- [ ] 理解Plan-and-Execute模式
- [ ] 掌握多Agent协作机制
- [ ] 实现自主规划Agent
- [ ] 完成研究助手Agent项目

### 下一步学习

- **下一章**：[第15章：知识图谱RAG](./15-知识图谱RAG.md)
- **相关资源**：
  - LangChain Agents: https://python.langchain.com/docs/modules/agents/
  - AutoGPT: https://github.com/Significant-Gravitas/AutoGPT

---

**恭喜完成第14章！** 🎉

**继续学习知识图谱RAG！** → [第15章](./15-知识图谱RAG.md)
