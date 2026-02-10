# 第4章：RAG评估基础

> 如何知道你的RAG系统好不好？本章将带你建立完整的评估体系，用科学的方法衡量和改进RAG系统。

---

## 📚 学习目标

学完本章后，你将能够：

- [ ] 理解RAG评估的重要性和评估维度
- [ ] 实现检索质量评估指标（Hit Rate、MRR）
- [ ] 实现生成质量评估指标（Faithfulness、Relevancy）
- [ ] 使用RAGAS评估框架
- [ ] 建立自动化的评估流程
- [ ] 基于评估结果优化RAG系统

**预计学习时间**：1.5小时
**难度等级**：⭐⭐☆☆☆

---

## 前置知识

在开始本章学习前，你需要：

- [ ] 完成第3章，有一个可运行的RAG系统
- [ ] 理解向量检索的基本原理
- [ ] 了解基础的统计指标概念

**环境要求**：
- 已安装RAG系统
- 需要额外的评估库：ragas

---

## 4.1 为什么需要评估？

### 评估的重要性

#### 问题：感觉好 ≠ 实际好

```
开发者视角：
  "我的RAG系统看起来工作正常！"
  → 偶尔测试几次，觉得还行

用户视角：
  "为什么总是找不到相关内容？"
  → 实际使用时频繁遇到问题
```

#### 需要科学的评估方法

**评估的目的**：

1. **量化性能**：用数字说话，不是靠感觉
2. **发现瓶颈**：找出系统的薄弱环节
3. **指导优化**：知道该改进什么
4. **对比方案**：客观比较不同技术
5. **持续监控**：确保系统稳定运行

### RAG评估的三个维度

```
┌─────────────────────────────────────────┐
│          RAG评估维度                     │
├─────────────────────────────────────────┤
│                                         │
│  1. 检索质量 (Retrieval Quality)       │
│     - 找到的文档是否相关？              │
│     - 相关文档是否排在前面？            │
│     指标：Hit Rate, MRR, Precision@K    │
│                                         │
│  2. 生成质量 (Generation Quality)      │
│     - 答案是否准确？                    │
│     - 是否基于检索文档？                │
│     指标：Faithfulness, Relevancy       │
│                                         │
│  3. 系统性能 (System Performance)      │
│     - 响应速度如何？                    │
│     - 能处理多少请求？                  │
│     指标：延迟, QPS, 成本               │
│                                         │
└─────────────────────────────────────────┘
```

---

## 4.2 检索质量评估

### Hit Rate（命中率）

#### 定义

**Hit Rate**：至少检索到一个相关文档的查询占比

```
Hit Rate = 有相关结果的查询数 / 总查询数
```

#### 计算

```python
# 文件名：04_01_hit_rate.py
"""
Hit Rate计算示例
"""

def calculate_hit_rate(queries, retrieved_docs, relevant_docs):
    """
    计算Hit Rate

    Args:
        queries: 查询列表
        retrieved_docs: 检索到的文档列表（每个查询的top-k结果）
        relevant_docs: 真实相关文档列表

    Returns:
        hit_rate: 命中率
    """
    hits = 0

    for query_id, retrieved in enumerate(retrieved_docs):
        # 获取该查询的真实相关文档
        relevant = set(relevant_docs[query_id])

        # 检查检索结果中是否有至少一个相关文档
        if any(doc_id in relevant for doc_id in retrieved):
            hits += 1

    hit_rate = hits / len(queries)
    return hit_rate

# 示例
if __name__ == "__main__":
    # 示例数据
    queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    # 检索结果（每个查询的top-3文档ID）
    retrieved_docs = [
        [1, 5, 8],    # Q1的结果
        [2, 6, 9],    # Q2的结果
        [3, 7, 10],   # Q3的结果
        [4, 8, 12],   # Q4的结果
        [5, 9, 13]    # Q5的结果
    ]

    # 真实相关文档
    relevant_docs = [
        {1, 8},      # Q1的相关文档
        {6},         # Q2的相关文档
        {11, 12},    # Q3的相关文档
        {4, 15},     # Q4的相关文档
        {9, 13}      # Q5的相关文档
    ]

    # 计算Hit Rate
    hit_rate = calculate_hit_rate(queries, retrieved_docs, relevant_docs)

    print("Hit Rate计算示例")
    print("="*50)
    for i, (retrieved, relevant) in enumerate(zip(retrieved_docs, relevant_docs), 1):
        hit = any(doc in relevant for doc in retrieved)
        status = "✓" if hit else "✗"
        print(f"Q{i}: {status} 检索={retrieved}, 相关={relevant}")

    print(f"\nHit Rate: {hit_rate:.2%}")
    print(f"解释: {hit_rate:.0%}的查询至少检索到一个相关文档")
```

#### 基准值

| Hit Rate | 评级 | 说明 |
|----------|------|------|
| < 0.5 | 差 | 需要大幅改进 |
| 0.5 - 0.7 | 中 | 基本可用 |
| 0.7 - 0.85 | 良好 | 表现不错 |
| > 0.85 | 优秀 | 达到生产标准 |

### MRR（Mean Reciprocal Rank）

#### 定义

**MRR**：平均倒数排名，衡量第一个相关文档的平均排名

```
MRR = (1 / 第一个相关文档的排名) 的平均值
```

#### 计算

```python
def calculate_mrr(retrieved_docs, relevant_docs):
    """
    计算MRR

    Args:
        retrieved_docs: 检索结果
        relevant_docs: 真实相关文档

    Returns:
        mrr: 平均倒数排名
    """
    reciprocal_ranks = []

    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        # 找到第一个相关文档的位置
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            # 没有找到相关文档
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

# 示例
if __name__ == "__main__":
    retrieved_docs = [
        [1, 5, 8],    # Q1：相关文档是1，排第1
        [2, 6, 9],    # Q2：相关文档是6，排第2
        [3, 7, 10],   # Q3：相关文档是11，不在结果中
        [4, 8, 12],   # Q4：相关文档是4，排第1
        [5, 9, 13]    # Q5：相关文档是13，排第3
    ]

    relevant_docs = [
        {1},         # Q1
        {6},         # Q2
        {11},        # Q3
        {4},         # Q4
        {13}         # Q5
    ]

    mrr = calculate_mrr(retrieved_docs, relevant_docs)

    print("\nMRR计算示例")
    print("="*50)
    print(f"Q1: 相关文档排第1 → 1/1 = 1.000")
    print(f"Q2: 相关文档排第2 → 1/2 = 0.500")
    print(f"Q3: 相关文档未检索到 → 0.000")
    print(f"Q4: 相关文档排第1 → 1/1 = 1.000")
    print(f"Q5: 相关文档排第3 → 1/3 = 0.333")
    print(f"\nMRR: (1.0 + 0.5 + 0.0 + 1.0 + 0.333) / 5 = {mrr:.3f}")
```

#### 基准值

| MRR | 评级 | 说明 |
|-----|------|------|
| < 0.3 | 差 | 相关文档排名靠后 |
| 0.3 - 0.5 | 中 | 相关文档在中游 |
| 0.5 - 0.7 | 良好 | 相关文档靠前 |
| > 0.7 | 优秀 | 相关文档通常在前3 |

### Precision@K

#### 定义

**Precision@K**：前K个结果中相关文档的比例

```python
def calculate_precision_at_k(retrieved_docs, relevant_docs, k=5):
    """
    计算Precision@K

    Args:
        retrieved_docs: 检索结果
        relevant_docs: 真实相关文档
        k: 前K个结果

    Returns:
        precision_at_k: 精确率
    """
    precisions = []

    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        # 取前K个结果
        top_k = retrieved[:k]

        # 计算相关文档数量
        relevant_count = sum(1 for doc in top_k if doc in relevant)

        # 精确率
        precision = relevant_count / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    return avg_precision
```

### 完整检索评估框架

```python
# 文件名：04_02_retrieval_eval.py
"""
完整的检索评估框架
"""

class RetrievalEvaluator:
    """检索质量评估器"""

    def __init__(self):
        self.metrics = {}

    def evaluate(self, queries, retrieved_docs, relevant_docs):
        """
        全面评估检索质量

        Args:
            queries: 查询列表
            retrieved_docs: 检索结果
            relevant_docs: 真实相关文档

        Returns:
            评估指标字典
        """
        results = {
            "hit_rate": self.calculate_hit_rate(retrieved_docs, relevant_docs),
            "mrr": self.calculate_mrr(retrieved_docs, relevant_docs),
            "precision_at_1": self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=1),
            "precision_at_3": self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=3),
            "precision_at_5": self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=5),
        }

        self.metrics = results
        return results

    def calculate_hit_rate(self, retrieved_docs, relevant_docs):
        """计算Hit Rate"""
        hits = sum(
            1 for retrieved, relevant in zip(retrieved_docs, relevant_docs)
            if any(doc in relevant for doc in retrieved)
        )
        return hits / len(retrieved_docs)

    def calculate_mrr(self, retrieved_docs, relevant_docs):
        """计算MRR"""
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k):
        """计算Precision@K"""
        precisions = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            top_k = retrieved[:k]
            relevant_count = sum(1 for doc in top_k if doc in relevant)
            precisions.append(relevant_count / k)
        return sum(precisions) / len(precisions)

    def print_report(self):
        """打印评估报告"""
        print("\n" + "="*60)
        print("检索质量评估报告")
        print("="*60)

        for metric, value in self.metrics.items():
            # 格式化指标名称
            metric_name = metric.replace("_", " ").title()
            # 格式化值
            if isinstance(value, float):
                print(f"{metric_name:20s}: {value:.3f}")
            else:
                print(f"{metric_name:20s}: {value}")

        print("="*60)

        # 评级
        hit_rate = self.metrics.get("hit_rate", 0)
        mrr = self.metrics.get("mrr", 0)

        if hit_rate > 0.85 and mrr > 0.7:
            rating = "优秀 ⭐⭐⭐⭐⭐"
        elif hit_rate > 0.7 and mrr > 0.5:
            rating = "良好 ⭐⭐⭐⭐"
        elif hit_rate > 0.5 and mrr > 0.3:
            rating = "中等 ⭐⭐⭐"
        else:
            rating = "需要改进 ⭐⭐"

        print(f"综合评级: {rating}")

# 使用示例
if __name__ == "__main__":
    # 示例数据
    queries = [f"Q{i}" for i in range(1, 11)]

    # 模拟检索结果（文档ID）
    retrieved_docs = [
        [1, 5, 8, 12, 15],
        [2, 6, 9, 13, 16],
        [3, 7, 10, 14, 17],
        [4, 8, 11, 15, 18],
        [5, 9, 12, 16, 19],
        [1, 6, 11, 16, 20],
        [2, 7, 12, 17, 21],
        [3, 8, 13, 18, 22],
        [4, 9, 14, 19, 23],
        [5, 10, 15, 20, 24]
    ]

    # 真实相关文档
    relevant_docs = [
        {1, 8},
        {6, 13},
        {3},
        {15},
        {9},
        {1, 11},
        {2, 17},
        {8, 13},
        {14},
        {20}
    ]

    # 评估
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(queries, retrieved_docs, relevant_docs)

    # 打印报告
    evaluator.print_report()
```

---

## 4.3 生成质量评估

### Faithfulness（忠实度）

#### 定义

**答案是否基于检索到的文档**，而不是LLM编造的。

#### 评估方法

```python
# 文件名：04_03_faithfulness.py
"""
Faithfulness评估示例
"""

from openai import OpenAI

def evaluate_faithfulness(answer, context_documents):
    """
    评估答案的忠实度

    Args:
        answer: RAG生成的答案
        context_documents: 检索到的上下文文档

    Returns:
        faithfulness_score: 忠实度分数 (0-1)
    """
    client = OpenAI()

    # 构建评估提示词
    context = "\n\n".join([
        f"文档{i+1}: {doc}"
        for i, doc in enumerate(context_documents)
    ])

    prompt = f"""
请评估以下答案是否基于提供的参考文档。

参考文档：
{context}

答案：
{answer}

请评估：
1. 答案中的所有声明是否都能在参考文档中找到支持？
2. 答案是否没有添加文档中不存在的信息？

评分标准：
- 1.0: 完全基于文档，无编造
- 0.7-0.9: 大部分基于文档，有少量合理推断
- 0.4-0.6: 部分基于文档，有明显添加信息
- 0.1-0.3: 大量编造信息
- 0.0: 完全不基于文档

请只返回一个0-1之间的分数，保留两位小数。
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个专业的评估助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # 提取分数
    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text)
        return max(0, min(1, score))  # 确保在0-1之间
    except ValueError:
        # 如果返回的不是纯数字，尝试提取
        import re
        numbers = re.findall(r'0\.\d+', score_text)
        if numbers:
            return float(numbers[0])
        return 0.5  # 默认分数

# 示例
if __name__ == "__main__":
    # 示例文档
    context = [
        "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
        "Python的特点是语法简洁、易学易用，适合初学者。"
    ]

    # 测试不同质量的答案
    test_cases = [
        {
            "name": "完全忠实",
            "answer": "Python由Guido van Rossum于1991年创建，它的特点是语法简洁、易学易用。"
        },
        {
            "name": "大部分忠实",
            "answer": "Python是一种高级编程语言，由Guido创建，它非常适合数据科学和Web开发。"
        },
        {
            "name": "部分编造",
            "answer": "Python是Google开发的编程语言，发布于2000年，主要用于人工智能领域。"
        },
        {
            "name": "完全编造",
            "answer": "Java是一种脚本语言，主要用于前端开发，语法非常复杂难学。"
        }
    ]

    print("Faithfulness评估示例")
    print("="*60 + "\n")

    for test in test_cases:
        score = evaluate_faithfulness(test["answer"], context)
        print(f"{test['name']:15s}: {score:.2f}")
        print(f"  答案: {test['answer']}")
        print()
```

### Relevancy（相关性）

#### 定义

**答案是否真正回答了用户的问题**。

#### 评估方法

```python
def evaluate_relevancy(question, answer):
    """
    评估答案的相关性

    Args:
        question: 用户问题
        answer: RAG生成的答案

    Returns:
        relevancy_score: 相关性分数 (0-1)
    """
    client = OpenAI()

    prompt = f"""
请评估以下答案是否真正回答了用户的问题。

用户问题：
{question}

答案：
{answer}

评估标准：
- 1.0: 完全回答了问题，信息充分准确
- 0.7-0.9: 很好地回答了问题，略有不足
- 0.4-0.6: 部分回答了问题，信息不完整
- 0.1-0.3: 基本没有回答问题
- 0.0: 完全不相关

请只返回一个0-1之间的分数，保留两位小数。
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个专业的评估助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text)
        return max(0, min(1, score))
    except ValueError:
        import re
        numbers = re.findall(r'0\.\d+', score_text)
        if numbers:
            return float(numbers[0])
        return 0.5
```

### 使用RAGAS评估框架

RAGAS是一个专门的RAG评估库，提供标准化的评估指标。

#### 安装RAGAS

```bash
pip install ragas
```

#### 使用RAGAS评估

```python
# 文件名：04_04_ragas_eval.py
"""
使用RAGAS框架评估
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from datasets import Dataset

def evaluate_with_ragas(questions, answers, contexts, ground_truths=None):
    """
    使用RAGAS评估

    Args:
        questions: 问题列表
        answers: RAG生成的答案列表
        contexts: 检索到的上下文列表
        ground_truths: 真实答案列表（可选）

    Returns:
        评估结果
    """
    # 准备数据
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    # 选择评估指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision
    ]

    # 运行评估
    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    return results

# 示例
if __name__ == "__main__":
    # 示例数据
    questions = [
        "Python是什么时候创建的？",
        "Python有什么特点？"
    ]

    answers = [
        "Python由Guido van Rossum于1991年创建。",
        "Python的特点是语法简洁、易学易用。"
    ]

    contexts = [
        ["Python是一种高级编程语言，由Guido van Rossum于1991年创建。"],
        ["Python的特点是语法简洁、易学易用，适合初学者。"]
    ]

    # 评估
    print("使用RAGAS评估")
    print("="*60 + "\n")

    results = evaluate_with_ragas(questions, answers, contexts)

    # 打印结果
    print(results.to_pandas())
```

---

## 4.4 建立评估基线

### 黄金数据集创建

#### 什么是黄金数据集？

人工标注的高质量问答对，作为评估的"标准答案"。

#### 创建步骤

```python
# 文件名：04_05_golden_dataset.py
"""
创建黄金数据集
"""

import json
from typing import List, Dict

class GoldenDatasetBuilder:
    """黄金数据集构建器"""

    def __init__(self, output_path="data/eval/golden_dataset.json"):
        self.output_path = output_path
        self.dataset = []

    def add_example(self, question: str, answer: str, context: str, metadata: dict = None):
        """
        添加一个示例

        Args:
            question: 问题
            answer: 标准答案
            context: 相关上下文
            metadata: 元数据
        """
        example = {
            "question": question,
            "answer": answer,
            "context": context,
            "metadata": metadata or {}
        }
        self.dataset.append(example)

    def save(self):
        """保存数据集"""
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"✓ 保存了 {len(self.dataset)} 个示例到 {self.output_path}")

    def load(self):
        """加载数据集"""
        with open(self.output_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        return self.dataset

# 创建示例数据集
if __name__ == "__main__":
    builder = GoldenDatasetBuilder()

    # 添加示例（基于我们的测试文档）
    examples = [
        {
            "question": "Python是什么时候创建的？",
            "answer": "Python由Guido van Rossum于1991年创建。",
            "context": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
            "metadata": {"topic": "Python历史", "difficulty": "easy"}
        },
        {
            "question": "Python有哪些特点？",
            "answer": "Python的特点是语法简洁、易学易用，适合初学者。",
            "context": "Python的特点是语法简洁、易学易用，适合初学者。",
            "metadata": {"topic": "Python特点", "difficulty": "easy"}
        },
        {
            "question": "JavaScript主要用于什么？",
            "answer": "JavaScript主要用于Web前端开发。",
            "context": "JavaScript是一种脚本语言，主要用于Web前端开发。",
            "metadata": {"topic": "JavaScript", "difficulty": "easy"}
        },
        {
            "question": "Rust如何保证内存安全？",
            "answer": "Rust通过所有权系统在编译时保证内存安全，不需要垃圾回收。",
            "context": "Rust是一种系统编程语言，注重内存安全、并发和性能。它没有垃圾回收，而是通过所有权系统在编译时保证内存安全。",
            "metadata": {"topic": "Rust", "difficulty": "medium"}
        }
    ]

    for example in examples:
        builder.add_example(**example)

    # 保存
    builder.save()

    print("\n黄金数据集创建完成！")
    print(f"包含 {len(builder.dataset)} 个问答对")
```

### 自动化评估流程

```python
# 文件名：04_06_auto_eval.py
"""
自动化RAG评估流程
"""

from golden_dataset import GoldenDatasetBuilder
from retrieval_eval import RetrievalEvaluator
from ragas_eval import evaluate_with_ragas
from rag_system import SimpleRAG

class AutoEvaluator:
    """自动化评估器"""

    def __init__(self, rag_system, golden_dataset_path):
        """
        初始化评估器

        Args:
            rag_system: RAG系统实例
            golden_dataset_path: 黄金数据集路径
        """
        self.rag = rag_system
        self.dataset = self.load_dataset(golden_dataset_path)

    def load_dataset(self, path):
        """加载黄金数据集"""
        builder = GoldenDatasetBuilder(path)
        return builder.load()

    def run_evaluation(self):
        """运行完整评估"""
        print("="*70)
        print("自动化RAG评估")
        print("="*70 + "\n")

        # 1. 准备数据
        questions = [item["question"] for item in self.dataset]
        ground_truths = [item["answer"] for item in self.dataset]

        # 2. 运行RAG系统
        print("步骤1: 运行RAG系统生成答案\n")
        rag_answers = []
        retrieved_contexts = []

        for question in questions:
            response = self.rag.query(question)
            rag_answers.append(str(response))
            # 假设response包含检索到的上下文
            retrieved_contexts.append([response.source_nodes[0].text])

        # 3. 评估生成质量
        print("步骤2: 评估生成质量（使用RAGAS）\n")
        generation_scores = evaluate_with_ragas(
            questions,
            rag_answers,
            retrieved_contexts,
            ground_truths
        )

        # 4. 评估检索质量
        print("步骤3: 评估检索质量\n")
        # 这里需要额外的真实相关文档标注
        # 简化示例：假设第一个检索到的文档是相关的
        retrieval_evaluator = RetrievalEvaluator()
        retrieval_scores = retrieval_evaluator.evaluate(
            questions,
            [[0] for _ in questions],  # 简化
            [set() for _ in questions]  # 简化
        )

        # 5. 综合报告
        self.print_report(generation_scores, retrieval_scores)

    def print_report(self, generation_scores, retrieval_scores):
        """打印综合报告"""
        print("\n" + "="*70)
        print("评估报告")
        print("="*70 + "\n")

        print("生成质量:")
        for metric, value in generation_scores.items():
            print(f"  {metric}: {value:.3f}")

        print("\n检索质量:")
        for metric, value in retrieval_scores.items():
            print(f"  {metric}: {value:.3f}")

        print("\n" + "="*70)

# 使用示例
if __name__ == "__main__":
    # 创建RAG系统
    rag = SimpleRAG()
    documents = rag.load_documents()
    rag.build_index(documents)

    # 运行评估
    evaluator = AutoEvaluator(rag, "data/eval/golden_dataset.json")
    evaluator.run_evaluation()
```

---

## 总结

### 本章要点回顾

1. **评估的重要性**：用科学方法衡量RAG系统
2. **检索质量评估**：Hit Rate、MRR、Precision@K
3. **生成质量评估**：Faithfulness、Relevancy
4. **RAGAS框架**：使用标准化工具评估
5. **建立评估基线**：创建黄金数据集和自动化流程

### 学习检查清单

- [ ] 理解RAG评估的三个维度
- [ ] 能够计算Hit Rate和MRR
- [ ] 掌握Faithfulness评估方法
- [ ] 会使用RAGAS框架
- [ ] 创建了黄金数据集
- [ ] 建立了自动化评估流程

### 下一步学习

- **下一章**：[第5章：模块1总结与项目](./05-模块1总结与项目.md)
  - 综合项目实战
  - 完整的端到端实现

### 扩展练习

1. **基础练习**：
   - 为你的RAG系统创建黄金数据集
   - 计算Hit Rate和MRR
   - 使用RAGAS评估

2. **进阶练习**：
   - 实现更多的评估指标
   - 可视化评估结果
   - 建立持续评估流程

3. **挑战项目**：
   - 对比不同分块策略的评估结果
   - 优化你的RAG系统直到达到优秀标准

---

## 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| **Hit Rate** | 命中率 | 至少检索到一个相关文档的查询占比 |
| **MRR** | Mean Reciprocal Rank | 平均倒数排名 |
| **Precision@K** | 精确率@K | 前K个结果中相关文档的比例 |
| **Faithfulness** | 忠实度 | 答案基于文档的程度 |
| **Relevancy** | 相关性 | 答案与问题的相关程度 |
| **黄金数据集** | Golden Dataset | 人工标注的高质量问答对 |

---

**返回目录** | **上一章：基础RAG实现** | **下一章：模块1总结与项目**

---

**本章结束**

> 评估是改进的基础。没有评估，我们就像在黑暗中摸索。现在你有了科学的评估方法，可以自信地衡量和优化你的RAG系统了！
