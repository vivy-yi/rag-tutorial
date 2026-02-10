# 案例2：技术文档问答系统

> 基于混合检索和重排序的技术文档搜索系统

## 快速开始

\`\`\`bash
pip install -r requirements.txt
cp .env.example .env
# 编辑.env添加API Key
streamlit run main.py
\`\`\`

## 功能特点

- ✅ 混合检索（向量 + BM25）
- ✅ CrossEncoder重排序
- ✅ 代码示例高亮
- ✅ 相关查询推荐

## 检索模式

1. **混合检索**：结合语义和关键词，效果最佳
2. **向量检索**：纯语义搜索，适合概念查询
3. **关键词检索**：精确匹配，适合代码搜索
