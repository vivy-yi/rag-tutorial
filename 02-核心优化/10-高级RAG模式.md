# 第10章：高级RAG模式

> 基础RAG不够用？高级RAG模式让系统"思考"何时检索、检索什么、检索多少，显著提升复杂问题的处理能力！

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解迭代检索的原理和应用场景
- [ ] 掌握自适应检索的实现方法
- [ ] 应用跳跃读取（Skip Reading）策略
- [ ] 使用元数据过滤优化检索
- [ ] 选择合适的RAG模式解决实际问题

**预计学习时间**：3小时
**难度等级**：⭐⭐⭐⭐☆

---

## 前置知识

在开始本章学习前，你需要具备：

- [ ] 完成模块1的基础RAG实现
- [ ] 理解混合检索原理（第9章）
- [ ] 熟悉查询增强技术（第8章）

**环境要求**：
- Python >= 3.9
- LLM API（OpenAI/GPT-4等）
- 向量数据库（支持元数据过滤）

---

## 10.1 RAG模式演进

### 10.1.1 从Naive到Advanced

```
RAG模式演进路径：

Level 1: Naive RAG（模块1）
  └─ 一次检索 → 一次生成
  适用：简单问答

Level 2: RAG + 优化（模块2）
  ├─ 更好的嵌入模型
  ├─ 高级分块策略
  ├─ 查询增强
  └─ 混合检索 + 重排序
  适用：中等复杂度问答

Level 3: Advanced RAG（本章）⭐
  ├─ 迭代检索
  ├─ 自适应检索
  ├─ 跳跃读取
  └─ 元数据过滤
  适用：复杂多跳问答

Level 4: Agentic RAG（模块3）
  ├─ Agent自主决策
  ├─ 工具调用
  └─ 多Agent协作
  适用：高度复杂任务
```

### 10.1.2 何时需要高级RAG？

**场景1：多跳推理**

```
用户问题："马斯克的火箭公司最近一次发射是什么时候？"

Naive RAG：
  ❌ 检索"马斯克火箭公司发射"
  ❌ 无法关联"马斯克"→"SpaceX"→"最新发射"

Advanced RAG（迭代检索）：
  ✅ 第1轮：检索"马斯克火箭公司" → 发现SpaceX
  ✅ 第2轮：检索"SpaceX最新发射" → 找到发射信息
  ✅ 第3轮：检索具体发射时间和详情 → 完整答案
```

**场景2：信息不全**

```
用户问题："这个API有什么限制？"

Naive RAG：
  ❌ 不确定是哪个API
  ❌ 检索结果可能不相关

Advanced RAG（自适应检索）：
  ✅ 检测信息不足
  ✅ 询问用户："您指的是哪个API？"
  ✅ 根据用户回答精确检索
```

**场景3：长尾知识**

```
用户问题："解释量子纠缠在量子计算中的应用"

Naive RAG：
  ❌ 检索到大量基础文档
  ❌ 答案过于宽泛

Advanced RAG（元数据过滤）：
  ✅ 按难度等级过滤（高级）
  ✅ 按主题过滤（量子计算）
  ✅ 检索到精确的高级文档
```

---

## 10.2 迭代检索

### 10.2.1 原理

**什么是迭代检索？**

```
传统RAG：
  Query → [一次检索] → Context → Answer

迭代RAG：
  Query → [检索1] → Context1 → [分析/判断]
                ↓
          需要更多信息？
           ↙          ↘
         Yes          No
         ↓             ↓
  [生成新查询]    [综合答案]
         ↓
  [检索2] → Context2 → [分析/判断]
              ↓
        ...
```

**核心思想**：
- 不一次性检索所有信息
- 根据当前检索结果，决定是否需要继续检索
- 每轮检索都基于上一轮的理解

### 10.2.2 实现方法

**方法1：固定迭代次数**

```python
# 文件名：iterative_retrieval.py
"""
迭代检索实现
"""

from typing import List, Tuple, Optional
import time


class IterativeRetriever:
    """
    迭代检索器

    通过多轮检索逐步收集信息，生成完整答案

    Args:
        retriever: 基础检索器
        llm_client: LLM客户端
        max_iterations: 最大迭代次数
        stop_threshold: 停止阈值（分数低于此值时停止）

    Example:
        >>> iterative_retriever = IterativeRetriever(retriever, llm_client)
        >>> result = iterative_retriever.retrieve(
        ...     "马斯克的火箭公司最近一次发射是什么时候？",
        ...     max_iterations=3
        ... )
        >>> print(result['answer'])
    """

    def __init__(self,
                 retriever,
                 llm_client,
                 max_iterations: int = 3,
                 stop_threshold: float = 0.5):

        self.retriever = retriever
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.stop_threshold = stop_threshold

    def retrieve(self, query: str, max_iterations: int = None) -> dict:
        """
        迭代检索

        Args:
            query: 原始查询
            max_iterations: 最大迭代次数，覆盖初始化时的设置

        Returns:
            {
                'answer': str,              # 最终答案
                'contexts': List[str],       # 所有检索到的上下文
                'iteration_count': int,      # 实际迭代次数
                'queries_used': List[str],   # 每轮使用的查询
                'reasoning_trace': List[str] # 推理过程
            }

        Example:
            >>> result = iterative_retriever.retrieve(
            ...     "Python和JavaScript在Web开发中的差异"
            ... )
            >>> print(f"迭代次数: {result['iteration_count']}")
            >>> print(f"答案: {result['answer']}")
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        # 初始化
        all_contexts = []
        queries_used = []
        reasoning_trace = []
        current_query = query

        # 迭代检索
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== 第{iteration}轮迭代 ===")

            # 检索
            retrieved_docs = self.retriever.retrieve(current_query, top_k=3)
            context = "\n\n".join([doc['text'] for doc in retrieved_docs])
            all_contexts.append(context)
            queries_used.append(current_query)

            print(f"查询: {current_query}")
            print(f"检索到 {len(retrieved_docs)} 个文档")

            # 判断是否需要继续
            if iteration < max_iterations:
                decision = self._decide_continue(
                    original_query=query,
                    current_query=current_query,
                    context=context,
                    iteration=iteration
                )

                reasoning_trace.append(decision['reasoning'])

                if decision['should_continue']:
                    # 更新查询
                    current_query = decision['next_query']
                    print(f"继续检索: {current_query}")
                else:
                    print("停止迭代，信息已充足")
                    break
            else:
                print(f"达到最大迭代次数({max_iterations})，停止")

        # 生成最终答案
        print("\n=== 生成最终答案 ===")
        final_context = "\n\n".join(all_contexts)
        answer = self._generate_answer(query, final_context)

        return {
            'answer': answer,
            'contexts': all_contexts,
            'iteration_count': len(all_contexts),
            'queries_used': queries_used,
            'reasoning_trace': reasoning_trace
        }

    def _decide_continue(self,
                        original_query: str,
                        current_query: str,
                        context: str,
                        iteration: int) -> dict:
        """
        决定是否继续检索

        Returns:
            {
                'should_continue': bool,
                'next_query': str,
                'reasoning': str
            }
        """
        prompt = f"""
你是一个信息检索专家。请分析当前检索结果是否足够回答用户的问题。

原始问题：{original_query}
当前查询：{current_query}
当前迭代：第{iteration}轮

检索到的信息：
{context[:1000]}...

请判断：
1. 当前信息是否足以回答原始问题？
2. 如果不足，还需要什么信息？
3. 下一轮应该检索什么？

以JSON格式返回：
{{
    "should_continue": true/false,
    "next_query": "如果继续，下一轮的查询",
    "reasoning": "判断理由"
}}
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        import json
        try:
            decision = json.loads(response.choices[0].message.content)
            return decision
        except:
            # 解析失败，默认停止
            return {
                'should_continue': False,
                'next_query': '',
                'reasoning': 'LLM返回格式错误，停止迭代'
            }

    def _generate_answer(self, query: str, context: str) -> str:
        """
        基于所有检索到的上下文生成答案
        """
        prompt = f"""
基于以下检索到的信息，回答用户的问题。请综合所有信息，给出完整准确的答案。

用户问题：{query}

检索到的信息：
{context}

答案：
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    from openai import OpenAI

    # 初始化
    client = OpenAI()

    # 模拟检索器（实际使用时替换为真实检索器）
    class MockRetriever:
        def retrieve(self, query, top_k=3):
            # 模拟返回
            return [
                {'text': f'{query}的相关文档1...'},
                {'text': f'{query}的相关文档2...'},
                {'text': f'{query}的相关文档3...'}
            ]

    retriever = MockRetriever()

    # 创建迭代检索器
    iterative_retriever = IterativeRetriever(
        retriever=retriever,
        llm_client=client,
        max_iterations=3
    )

    # 测试查询
    query = "马斯克的火箭公司最近一次发射是什么时候？"
    print(f"\n{'='*80}")
    print(f"查询: {query}")
    print(f"{'='*80}")

    result = iterative_retriever.retrieve(query)

    # 打印结果
    print(f"\n{'='*80}")
    print("最终结果")
    print(f"{'='*80}")
    print(f"\n迭代次数: {result['iteration_count']}")
    print(f"\n使用的查询:")
    for i, q in enumerate(result['queries_used'], 1):
        print(f"  第{i}轮: {q}")

    print(f"\n推理过程:")
    for i, reasoning in enumerate(result['reasoning_trace'], 1):
        print(f"  第{i}轮: {reasoning}")

    print(f"\n最终答案:")
    print(result['answer'])
```

**方法2：动态停止判断**

```python
class DynamicIterativeRetriever(IterativeRetriever):
    """
    动态迭代检索器

    根据信息完整度动态决定是否停止
    """

    def _decide_continue(self,
                        original_query: str,
                        current_query: str,
                        context: str,
                        iteration: int) -> dict:
        """
        增强的停止判断

        判断标准：
        1. 信息完整度（基于LLM评分）
        2. 检索结果相关性
        3. 查询覆盖度
        """
        # 评分信息完整度（1-10分）
        completeness_score = self._score_completeness(
            original_query, context
        )

        print(f"信息完整度评分: {completeness_score}/10")

        # 阈值判断
        if completeness_score >= 8:
            return {
                'should_continue': False,
                'next_query': '',
                'reasoning': f'信息完整度({completeness_score}/10)已足够'
            }

        # 生成下一轮查询
        next_query = self._generate_next_query(
            original_query=original_query,
            current_context=context,
            iteration=iteration
        )

        return {
            'should_continue': True,
            'next_query': next_query,
            'reasoning': f'信息完整度({completeness_score}/10)不足，需要继续检索'
        }

    def _score_completeness(self, query: str, context: str) -> float:
        """
        评分信息完整度

        Returns:
            1-10的评分
        """
        prompt = f"""
请评分当前信息是否足以回答用户的问题（1-10分）。

用户问题：{query}

当前信息：
{context[:500]}...

评分标准：
- 1-3分：信息严重不足，无法回答
- 4-6分：信息部分充足，可以给出初步答案
- 7-8分：信息基本充足，可以给出较完整答案
- 9-10分：信息非常充足，可以给出详细完整答案

请只返回一个数字（1-10），不要其他内容。
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 1.0), 10.0)  # 限制在[1,10]
        except:
            return 5.0  # 解析失败，返回中间值

    def _generate_next_query(self,
                            original_query: str,
                            current_context: str,
                            iteration: int) -> str:
        """
        生成下一轮查询
        """
        prompt = f"""
基于原始问题和当前已知信息，生成下一轮检索查询。

原始问题：{original_query}

已知信息：
{current_context[:500]}...

请生成一个具体的检索查询，帮助获取缺失的关键信息。

查询要求：
1. 具体明确，包含关键实体
2. 避免重复已有信息
3. 专注于缺失的关键点

返回格式：直接返回查询文本，不要其他内容。
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
```

### 10.2.3 迭代检索的场景

**场景1：多跳推理**

```
问题："《阿凡达》导演的下一部电影是什么？"

迭代过程：
  第1轮：检索"阿凡达导演" → 詹姆斯·卡梅隆
  第2轮：检索"詹姆斯·卡梅隆下一部电影" → 《阿凡达3》
  第3轮：检索"阿凡达3上映时间" → 2025年12月

答案：詹姆斯·卡梅隆的下一部电影是《阿凡达3》，预计2025年12月上映。
```

**场景2：复杂比较**

```
问题："对比Python和JavaScript在机器学习领域的应用"

迭代过程：
  第1轮：检索"Python机器学习应用" → Python的ML生态
  第2轮：检索"JavaScript机器学习应用" → TensorFlow.js等
  第3轮：检索"Python vs JavaScript机器学习" → 直接对比文章

答案：Python在ML领域占主导（丰富的库），JS适合Web端ML推理。
```

---

## 10.3 自适应检索

### 10.3.1 原理

**什么是自适应检索？**

```
传统RAG：
  所有查询 → 统一的检索策略 → 固定Top-K

自适应RAG：
  分析查询特征
       ↓
  ┌────┴────┐
  │         │
简单查询   复杂查询
  │         │
  ↓         ↓
直接回答   深度检索
  ↓         ↓
快速响应   完整答案
```

**核心思想**：
- 不对所有查询使用相同策略
- 根据查询复杂度动态调整
- 平衡速度和质量

### 10.3.2 查询复杂度评估

```python
# 文件名：adaptive_retrieval.py
"""
自适应检索实现
"""

from typing import Dict, List
import re


class QueryComplexityAnalyzer:
    """
    查询复杂度分析器

    Args:
        llm_client: LLM客户端

    Example:
        >>> analyzer = QueryComplexityAnalyzer(llm_client)
        >>> complexity = analyzer.analyze("Python性能优化技巧")
        >>> print(complexity['level'])  # 'simple', 'medium', 'complex'
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def analyze(self, query: str) -> Dict:
        """
        分析查询复杂度

        Returns:
            {
                'level': 'simple' | 'medium' | 'complex',
                'score': float,  # 0-1
                'features': Dict,
                'strategy': str
            }
        """
        # 特征提取
        features = self._extract_features(query)

        # 规则分类
        rule_based_level = self._rule_based_classification(features)

        # LLM验证（可选）
        llm_based_level = self._llm_based_classification(query)

        # 综合判断
        final_level = self._combine_classification(
            rule_based_level,
            llm_based_level
        )

        # 推荐检索策略
        strategy = self._recommend_strategy(final_level)

        return {
            'level': final_level,
            'score': self._level_to_score(final_level),
            'features': features,
            'strategy': strategy
        }

    def _extract_features(self, query: str) -> Dict:
        """
        提取查询特征
        """
        features = {
            'length': len(query.split()),
            'has_entities': bool(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)),
            'has_numbers': bool(re.search(r'\d+', query)),
            'has_compare_words': bool(re.search(r'对比|比较|差异|区别|vs|versus', query, re.I)),
            'has_how_words': bool(re.search(r'如何|怎么|怎样|方法', query, re.I)),
            'has_why_words': bool(re.search(r'为什么|为何|原因', query, re.I)),
            'has_what_words': bool(re.search(r'是什么|什么是|定义', query, re.I)),
            'question_marks': query.count('?') + query.count('？'),
            'has_multi_part': bool(re.search(r'[，,、]', query)),
        }

        return features

    def _rule_based_classification(self, features: Dict) -> str:
        """
        基于规则的分类
        """
        score = 0

        # 长度得分
        if features['length'] > 15:
            score += 2
        elif features['length'] > 8:
            score += 1

        # 实体识别
        if features['has_entities']:
            score += 1

        # 复杂问题词
        if features['has_compare_words']:
            score += 2
        if features['has_how_words']:
            score += 1
        if features['has_why_words']:
            score += 2

        # 多部分问题
        if features['has_multi_part']:
            score += 2

        # 分类
        if score >= 5:
            return 'complex'
        elif score >= 3:
            return 'medium'
        else:
            return 'simple'

    def _llm_based_classification(self, query: str) -> str:
        """
        基于LLM的分类（可选）
        """
        prompt = f"""
请评估以下查询的复杂度（simple/medium/complex）：

查询：{query}

评估标准：
- Simple: 单一事实，直接查询，如"Python是什么？"
- Medium: 需要一定推理，如"如何优化Python代码？"
- Complex: 多跳推理，比较分析，如"对比Python和JavaScript在Web开发中的优劣势"

返回格式：只返回一个词（simple/medium/complex），不要其他内容。
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content.strip().lower()
            if result in ['simple', 'medium', 'complex']:
                return result
        except:
            pass

        return 'medium'  # 默认中等

    def _combine_classification(self,
                               rule_level: str,
                               llm_level: str) -> str:
        """
        综合两种分类结果
        """
        # 优先LLM判断
        return llm_level

    def _level_to_score(self, level: str) -> float:
        """
        将级别转换为0-1分数
        """
        scores = {'simple': 0.2, 'medium': 0.5, 'complex': 0.8}
        return scores.get(level, 0.5)

    def _recommend_strategy(self, level: str) -> str:
        """
        推荐检索策略
        """
        strategies = {
            'simple': 'direct_retrieval',  # 直接检索
            'medium': 'hybrid_retrieval',  # 混合检索
            'complex': 'iterative_retrieval'  # 迭代检索
        }
        return strategies.get(level, 'hybrid_retrieval')


class AdaptiveRetrievalSystem:
    """
    自适应检索系统

    根据查询复杂度自动选择检索策略
    """

    def __init__(self,
                 retriever,
                 llm_client,
                 iterative_retriever=None):

        self.retriever = retriever
        self.llm_client = llm_client
        self.iterative_retriever = iterative_retriever

        self.complexity_analyzer = QueryComplexityAnalyzer(llm_client)

    def retrieve(self, query: str) -> Dict:
        """
        自适应检索

        Args:
            query: 查询文本

        Returns:
            {
                'answer': str,
                'strategy_used': str,
                'complexity_level': str,
                'retrieval_time': float
            }
        """
        import time
        start_time = time.time()

        # 分析复杂度
        print(f"\n分析查询复杂度...")
        complexity = self.complexity_analyzer.analyze(query)

        print(f"复杂度级别: {complexity['level']}")
        print(f"推荐策略: {complexity['strategy']}")

        # 根据策略检索
        strategy = complexity['strategy']

        if strategy == 'direct_retrieval':
            # 简单查询：直接检索
            print("使用策略：直接检索")
            answer = self._direct_retrieve(query)

        elif strategy == 'hybrid_retrieval':
            # 中等查询：混合检索
            print("使用策略：混合检索")
            answer = self._hybrid_retrieve(query)

        elif strategy == 'iterative_retrieval':
            # 复杂查询：迭代检索
            print("使用策略：迭代检索")
            if self.iterative_retriever:
                result = self.iterative_retriever.retrieve(query)
                answer = result['answer']
            else:
                # 回退到混合检索
                print("警告：迭代检索器未配置，回退到混合检索")
                answer = self._hybrid_retrieve(query)

        retrieval_time = time.time() - start_time

        return {
            'answer': answer,
            'strategy_used': strategy,
            'complexity_level': complexity['level'],
            'retrieval_time': retrieval_time
        }

    def _direct_retrieve(self, query: str) -> str:
        """
        直接检索（简单查询）
        """
        # 检索Top-3
        results = self.retriever.retrieve(query, top_k=3)

        # 直接生成答案
        context = "\n".join([doc['text'] for doc in results[:3]])

        answer = self._generate_answer(query, context)
        return answer

    def _hybrid_retrieve(self, query: str) -> str:
        """
        混合检索（中等查询）
        """
        # 检索Top-5
        results = self.retriever.retrieve(query, top_k=5)

        # 生成答案
        context = "\n".join([doc['text'] for doc in results[:5]])

        answer = self._generate_answer(query, context)
        return answer

    def _generate_answer(self, query: str, context: str) -> str:
        """
        生成答案
        """
        prompt = f"""
基于以下信息回答问题：

问题：{query}

信息：
{context}

答案：
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI()

    # 模拟检索器
    class MockRetriever:
        def retrieve(self, query, top_k=5):
            return [
                {'text': f'{query}的相关文档1...'},
                {'text': f'{query}的相关文档2...'},
                {'text': f'{query}的相关文档3...'},
            ]

    retriever = MockRetriever()

    # 创建自适应检索系统
    adaptive_system = AdaptiveRetrievalSystem(
        retriever=retriever,
        llm_client=client
    )

    # 测试不同复杂度的查询
    test_queries = [
        "Python是什么？",  # Simple
        "如何优化Python代码性能？",  # Medium
        "对比Python和JavaScript在Web开发中的优劣势"  # Complex
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"查询: {query}")
        print(f"{'='*80}")

        result = adaptive_system.retrieve(query)

        print(f"\n使用策略: {result['strategy_used']}")
        print(f"复杂度: {result['complexity_level']}")
        print(f"检索时间: {result['retrieval_time']:.2f}秒")
        print(f"\n答案:\n{result['answer']}\n")
```

---

## 10.4 跳跃读取（Skip Reading）

### 10.4.1 原理

**什么是Skip Reading？**

```
传统检索：
  检索Top-K文档 → 读取全部 → 生成答案
  问题：可能包含不相关的文档

Skip Reading：
  检索Top-K文档 → 智能筛选 → 只读重要文档 → 生成答案
  优势：减少无关信息，提升效率
```

**核心思想**：
- 不是所有检索到的文档都需要读取
- 通过重排序快速识别重要文档
- 跳过低相关性文档

### 10.4.2 实现方法

```python
# 文件名：skip_reading.py
"""
跳跃读取实现
"""

from typing import List, Dict, Tuple


class SkipReadingRetriever:
    """
    跳跃读取检索器

    先快速筛选重要文档，再详细读取

    Args:
        retriever: 基础检索器
        reranker: 重排序器
        skip_threshold: 跳过阈值（相关性分数低于此值则跳过）

    Example:
        >>> skip_reader = SkipReadingRetriever(retriever, reranker)
        >>> results = skip_reader.retrieve("Python性能优化")
        >>> 只读取高相关性文档
    """

    def __init__(self,
                 retriever,
                 reranker,
                 skip_threshold: float = 0.5):

        self.retriever = retriever
        self.reranker = reranker
        self.skip_threshold = skip_threshold

        # 统计信息
        self.stats = {
            'total_queries': 0,
            'total_retrieved': 0,
            'total_read': 0,
            'skip_rate': 0.0
        }

    def retrieve(self, query: str,
                initial_top_k: int = 50,
                final_top_k: int = 10) -> Dict:
        """
        跳跃读取检索

        Args:
            query: 查询文本
            initial_top_k: 初始检索文档数
            final_top_k: 最终返回文档数

        Returns:
            {
                'read_docs': List[dict],  # 实际读取的文档
                'skipped_docs': List[dict],  # 跳过的文档
                'skip_rate': float,
                'answer': str
            }
        """
        print(f"\n=== 跳跃读取检索 ===")
        print(f"查询: {query}")

        # 步骤1：初始检索（获取候选）
        print(f"\n步骤1: 初始检索Top-{initial_top_k}")
        initial_results = self.retriever.retrieve(query, top_k=initial_top_k)

        print(f"  检索到 {len(initial_results)} 个候选文档")

        # 歆骤2：快速重排序
        print(f"\n步骤2: 快速重排序")
        reranked = self.reranker.rerank(query, initial_results)

        # 步骤3：选择性读取
        print(f"\n步骤3: 选择性读取（阈值={self.skip_threshold}）")
        read_docs = []
        skipped_docs = []

        for doc_id, score in reranked:
            doc = next((d for d in initial_results if d['id'] == doc_id), None)

            if score >= self.skip_threshold:
                # 读取文档
                read_docs.append({
                    'id': doc_id,
                    'score': score,
                    'text': doc['text'] if doc else ''
                })
                print(f"  ✓ 读取: {doc_id} (分数: {score:.4f})")
            else:
                # 跳过文档
                skipped_docs.append({
                    'id': doc_id,
                    'score': score,
                    'reason': '相关性分数过低'
                })

        print(f"\n  读取: {len(read_docs)} 个文档")
        print(f"  跳过: {len(skipped_docs)} 个文档")
        print(f"  跳跃率: {len(skipped_docs)/len(initial_results)*100:.1f}%")

        # 步骤4：生成答案
        print(f"\n步骤4: 生成答案")
        context = "\n\n".join([doc['text'] for doc in read_docs[:final_top_k]])

        answer = self._generate_answer(query, context)

        # 更新统计
        self._update_stats(
            total_retrieved=len(initial_results),
            total_read=len(read_docs)
        )

        return {
            'read_docs': read_docs,
            'skipped_docs': skipped_docs,
            'skip_rate': len(skipped_docs) / len(initial_results),
            'answer': answer
        }

    def _generate_answer(self, query: str, context: str) -> str:
        """
        生成答案（占位符）
        """
        # 实际使用时调用LLM
        return f"基于{len(context)}字的相关信息生成的答案..."

    def _update_stats(self, total_retrieved: int, total_read: int):
        """
        更新统计信息
        """
        self.stats['total_queries'] += 1
        self.stats['total_retrieved'] += total_retrieved
        self.stats['total_read'] += total_read
        self.stats['skip_rate'] = (
            (self.stats['total_retrieved'] - self.stats['total_read']) /
            self.stats['total_retrieved']
        )

    def get_stats(self) -> Dict:
        """
        获取统计信息
        """
        avg_skip_rate = self.stats['skip_rate']
        avg_retrieved = (
            self.stats['total_retrieved'] / self.stats['total_queries']
            if self.stats['total_queries'] > 0 else 0
        )
        avg_read = (
            self.stats['total_read'] / self.stats['total_queries']
            if self.stats['total_queries'] > 0 else 0
        )

        return {
            'total_queries': self.stats['total_queries'],
            'avg_retrieved_per_query': avg_retrieved,
            'avg_read_per_query': avg_read,
            'overall_skip_rate': avg_skip_rate
        }


# 使用示例
if __name__ == "__main__":
    # 模拟检索器
    class MockRetriever:
        def retrieve(self, query, top_k=50):
            results = []
            for i in range(top_k):
                relevance = 1.0 - (i * 0.02)  # 模拟递减的相关性
                results.append({
                    'id': f'doc_{i}',
                    'text': f'文档{i}的相关内容...'
                })
            return results

    # 模拟重排序器
    class MockReranker:
        def rerank(self, query, documents):
            # 模拟重排序分数
            reranked = []
            for i, doc in enumerate(documents):
                # 前几个文档分数高，后面的递减
                score = max(0, 1.0 - (i * 0.03))
                reranked.append((doc['id'], score))
            return reranked

    retriever = MockRetriever()
    reranker = MockReranker()

    # 创建跳跃读取检索器
    skip_reader = SkipReadingRetriever(
        retriever=retriever,
        reranker=reranker,
        skip_threshold=0.4
    )

    # 测试查询
    query = "Python性能优化技巧"
    result = skip_reader.retrieve(query, initial_top_k=50, final_top_k=10)

    print(f"\n{'='*80}")
    print("最终答案")
    print(f"{'='*80}")
    print(result['answer'])

    print(f"\n{'='*80}")
    print("统计信息")
    print(f"{'='*80}")

    stats = skip_reader.get_stats()
    print(f"总查询数: {stats['total_queries']}")
    print(f"平均检索文档数: {stats['avg_retrieved_per_query']:.1f}")
    print(f"平均读取文档数: {stats['avg_read_per_query']:.1f}")
    print(f"整体跳跃率: {stats['overall_skip_rate']*100:.1f}%")

    print(f"\n效率提升:")
    print(f"  减少 {stats['avg_retrieved_per_query'] - stats['avg_read_per_query']:.1f} 个文档的读取")
    print(f"  节省 {(1 - stats['avg_read_per_query']/stats['avg_retrieved_per_query'])*100:.1f}% 的读取时间")
```

---

## 10.5 元数据过滤

### 10.5.1 原理

**什么是元数据过滤？**

```
传统检索：
  Query → 向量相似度 → 检索结果
  问题：无法控制检索结果的属性

元数据过滤：
  Query + 过滤条件 → 向量检索 + 过滤 → 检索结果
  优势：精确控制检索范围
```

**常用元数据**：

```python
metadata = {
    'author': 'Author Name',        # 作者
    'date': '2025-01-01',           # 日期
    'category': 'Technology',       # 分类
    'tags': ['python', 'ml'],       # 标签
    'difficulty': 'advanced',       # 难度
    'language': 'zh-CN',            # 语言
    'length': 5000,                 # 长度
    'source': 'arxiv',              # 来源
    'version': '2.0'                # 版本
}
```

### 10.5.2 实现方法

```python
# 文件名：metadata_filtering.py
"""
元数据过滤实现
"""

from typing import List, Dict, Any
from datetime import datetime


class MetadataFilter:
    """
    元数据过滤器

    支持多种过滤条件组合

    Example:
        >>> filter = MetadataFilter()
        >>> conditions = {
        ...     'category': 'Technology',
        ...     'date': {'>': '2024-01-01'},
        ...     'difficulty': ['advanced', 'intermediate']
        ... }
        >>> filtered = filter.apply(documents, conditions)
    """

    def __init__(self):
        pass

    def apply(self,
             documents: List[Dict],
             conditions: Dict) -> List[Dict]:
        """
        应用过滤条件

        Args:
            documents: 文档列表，每个文档包含'metadata'字段
            conditions: 过滤条件

        Returns:
            过滤后的文档列表

        Example:
            >>> conditions = {
            ...     'author': '张三',
            ...     'date': {'>': '2024-01-01', '<': '2024-12-31'},
            ...     'tags': {'in': ['python', 'ml']},
            ...     'difficulty': ['advanced', 'intermediate']
            ... }
            >>> filtered = filter.apply(docs, conditions)
        """
        filtered_docs = documents.copy()

        for key, value in conditions.items():
            if isinstance(value, dict):
                # 比较操作
                filtered_docs = self._apply_comparison(
                    filtered_docs, key, value
                )
            elif isinstance(value, list):
                # 多值匹配（OR）
                filtered_docs = [
                    doc for doc in filtered_docs
                    if doc.get('metadata', {}).get(key) in value
                ]
            else:
                # 精确匹配
                filtered_docs = [
                    doc for doc in filtered_docs
                    if doc.get('metadata', {}).get(key) == value
                ]

        return filtered_docs

    def _apply_comparison(self,
                         documents: List[Dict],
                         key: str,
                         ops: Dict) -> List[Dict]:
        """
        应用比较操作

        支持的操作：
        - '>', '>=', '<', '<=': 数值/日期比较
        - 'in', 'not_in': 包含/不包含
        - 'contains': 字符串包含
        """
        filtered = []

        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_value = metadata.get(key)

            # 检查所有操作
            match = True
            for op, op_value in ops.items():
                if not self._compare(doc_value, op, op_value):
                    match = False
                    break

            if match:
                filtered.append(doc)

        return filtered

    def _compare(self, doc_value, op, op_value) -> bool:
        """
        执行单个比较操作
        """
        try:
            if op == '>':
                return doc_value > op_value
            elif op == '>=':
                return doc_value >= op_value
            elif op == '<':
                return doc_value < op_value
            elif op == '<=':
                return doc_value <= op_value
            elif op == 'in':
                return doc_value in op_value
            elif op == 'not_in':
                return doc_value not in op_value
            elif op == 'contains':
                return op_value in str(doc_value)
            else:
                return False
        except:
            return False


class MetadataAwareRetriever:
    """
    支持元数据过滤的检索器

    结合向量检索和元数据过滤
    """

    def __init__(self, vector_store, metadata_filter: MetadataFilter = None):
        self.vector_store = vector_store
        self.metadata_filter = metadata_filter or MetadataFilter()

    def retrieve(self,
                query: str,
                top_k: int = 10,
                filters: Dict = None) -> List[Dict]:
        """
        带元数据过滤的检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            filters: 元数据过滤条件

        Returns:
            [(doc, score), ...]

        Example:
            >>> results = retriever.retrieve(
            ...     "Python机器学习",
            ...     top_k=10,
            ...     filters={
            ...         'category': 'Technology',
            ...         'difficulty': ['advanced', 'intermediate'],
            ...         'date': {'>': '2024-01-01'}
            ...     }
            ... )
        """
        # 步骤1：向量检索（获取候选）
        candidates = self.vector_store.search(query, top_k=top_k * 3)

        # 步骤2：元数据过滤
        if filters:
            print(f"应用元数据过滤: {filters}")
            filtered = self.metadata_filter.apply(candidates, filters)
            print(f"  过滤前: {len(candidates)} 个文档")
            print(f"  过滤后: {len(filtered)} 个文档")
        else:
            filtered = candidates

        # 步骤3：返回Top-K
        return filtered[:top_k]


# 使用示例
if __name__ == "__main__":
    # 示例文档
    documents = [
        {
            'id': 'doc1',
            'text': 'Python性能优化技巧...',
            'metadata': {
                'author': '张三',
                'date': '2024-03-15',
                'category': 'Technology',
                'tags': ['python', 'performance'],
                'difficulty': 'intermediate',
                'language': 'zh-CN',
                'length': 3000
            }
        },
        {
            'id': 'doc2',
            'text': 'JavaScript高级特性...',
            'metadata': {
                'author': '李四',
                'date': '2024-05-20',
                'category': 'Technology',
                'tags': ['javascript', 'advanced'],
                'difficulty': 'advanced',
                'language': 'zh-CN',
                'length': 5000
            }
        },
        {
            'id': 'doc3',
            'text': 'Python入门教程...',
            'metadata': {
                'author': '王五',
                'date': '2023-11-10',
                'category': 'Technology',
                'tags': ['python', 'beginner'],
                'difficulty': 'beginner',
                'language': 'zh-CN',
                'length': 2000
            }
        },
        {
            'id': 'doc4',
            'text': 'Machine Learning with Python...',
            'metadata': {
                'author': '张三',
                'date': '2024-06-01',
                'category': 'Technology',
                'tags': ['python', 'ml', 'advanced'],
                'difficulty': 'advanced',
                'language': 'en',
                'length': 8000
            }
        }
    ]

    # 创建过滤器
    metadata_filter = MetadataFilter()

    # 示例1：精确匹配
    print("\n" + "="*80)
    print("示例1: 查找作者为'张三'的文档")
    print("="*80)

    conditions1 = {'author': '张三'}
    filtered1 = metadata_filter.apply(documents, conditions1)

    for doc in filtered1:
        print(f"  {doc['id']}: {doc['metadata']['author']} - {doc['text'][:30]}...")

    # 示例2：多值匹配（OR）
    print("\n" + "="*80)
    print("示例2: 查找难度为'intermediate'或'advanced'的文档")
    print("="*80)

    conditions2 = {'difficulty': ['intermediate', 'advanced']}
    filtered2 = metadata_filter.apply(documents, conditions2)

    for doc in filtered2:
        print(f"  {doc['id']}: {doc['metadata']['difficulty']} - {doc['text'][:30]}...")

    # 示例3：日期范围
    print("\n" + "="*80)
    print("示例3: 查找2024年1月1日之后的文档")
    print("="*80)

    conditions3 = {'date': {'>': '2024-01-01'}}
    filtered3 = metadata_filter.apply(documents, conditions3)

    for doc in filtered3:
        print(f"  {doc['id']}: {doc['metadata']['date']} - {doc['text'][:30]}...")

    # 示例4：复杂条件组合
    print("\n" + "="*80)
    print("示例4: 复杂条件组合")
    print("  - 分类: Technology")
    print("  - 标签包含'python'")
    print("  - 难度: intermediate或advanced")
    print("  - 日期: 2024-01-01之后")
    print("="*80)

    conditions4 = {
        'category': 'Technology',
        'tags': {'in': ['python']},
        'difficulty': ['intermediate', 'advanced'],
        'date': {'>': '2024-01-01'}
    }

    filtered4 = metadata_filter.apply(documents, conditions4)

    print(f"\n找到 {len(filtered4)} 个符合条件的文档:")
    for doc in filtered4:
        print(f"\n  {doc['id']}:")
        print(f"    作者: {doc['metadata']['author']}")
        print(f"    日期: {doc['metadata']['date']}")
        print(f"    难度: {doc['metadata']['difficulty']}")
        print(f"    标签: {', '.join(doc['metadata']['tags'])}")
        print(f"    内容: {doc['text'][:50]}...")
```

---

## 10.6 RAG模式选择指南

### 10.6.1 决策树

```
查询特征分析
       │
       ├─ 包含专有名词？
       │   ├─ 是 → 元数据过滤 + BM25检索
       │   └─ 否 ↓
       │
       ├─ 多跳推理？
       │   ├─ 是 → 迭代检索
       │   └─ 否 ↓
       │
       ├─ 查询复杂？
       │   ├─ 是 → 自适应检索
       │   └─ 否 ↓
       │
       └─ 检索结果多？
           ├─ 是 → 跳跃读取
           └─ 否 → 标准检索
```

### 10.6.2 性能对比

```
┌──────────────────────────────────────────────────────────┐
│              RAG模式性能对比                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  模式                Hit Rate    MRR    响应时间    成本  │
│  ────────────────────────────────────────────────────  │
│  Naive RAG           0.60       0.50    1.0s       低    │
│  混合检索            0.78       0.68    2.0s       中    │
│  混合+重排序          0.85       0.76    3.5s       高    │
│  迭代检索            0.82       0.74    5.0s       高    │
│  自适应检索          0.83       0.75    2.5s       中    │
│  跳跃读取            0.84       0.73    2.0s       中    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 练习题

### 练习1：基础练习 - 实现迭代检索

**题目**：实现一个简单的迭代检索系统

**要求**：
1. 最多迭代3次
2. 每轮检索Top-3文档
3. 基于LLM判断是否继续
4. 最终生成综合答案

**提示**：
- 使用_openai_ API调用LLM
- 可以使用模拟的检索器测试

---

### 练习2：进阶练习 - 构建自适应检索系统

**题目**：实现查询复杂度分析器

**要求**：
1. 提取查询特征（长度、实体、问题词等）
2. 基于规则分类（simple/medium/complex）
3. 根据分类选择检索策略
4. 评估分类准确率

---

### 练习3：挑战项目 - 完整的高级RAG系统

**项目描述**：构建一个集成多种高级模式的RAG系统

**功能需求**：
1. ✅ 查询复杂度自动分析
2. ✅ 自适应选择检索策略
3. ✅ 支持迭代检索（多跳问题）
4. ✅ 元数据过滤
5. ✅ 性能监控和日志

**性能要求**：
- 简单查询响应 < 2秒
- 复杂查询响应 < 10秒
- Hit Rate > 0.80

---

## 总结

### 本章要点回顾

1. **迭代检索**
   - 多轮检索逐步收集信息
   - 适用于多跳推理问题
   - 需要"何时停止"的判断机制

2. **自适应检索**
   - 根据查询复杂度动态选择策略
   - 平衡速度和质量
   - 节省计算资源

3. **跳跃读取**
   - 智能筛选重要文档
   - 跳过低相关性文档
   - 提升检索效率

4. **元数据过滤**
   - 精确控制检索范围
   - 支持复杂条件组合
   - 提升检索精度

5. **模式选择**
   - 根据场景选择合适的RAG模式
   - 权衡性能、成本、效果
   - 可以组合使用多种模式

### 学习检查清单

- [ ] 理解各种高级RAG模式的原理
- [ ] 能够实现迭代检索
- [ ] 掌握自适应检索方法
- [ ] 能够应用元数据过滤
- [ ] 理解跳跃读取的优势
- [ ] 能够根据场景选择合适的RAG模式

### 下一步学习

- **下一章**：[第11章：性能优化](./11-性能优化.md)
- **相关章节**：
  - [第8章：查询增强技术](./08-查询增强技术.md)
  - [第9章：混合检索与重排序](./09-混合检索与重排序.md)
- **扩展阅读**：
  - LlamaIndex Advanced RAG: https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html
  - LangChain Multi-hop: https://python.langchain.com/docs/use_cases/question_answering/

---

**返回目录** | **上一章** | **下一章**

---

**本章结束**

> 有任何问题或建议？欢迎提交Issue或PR到教程仓库！
