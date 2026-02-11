# 案例1：智能客服RAG系统

> **难度**: ⭐ 入门 | **技术栈**: LangChain, OpenAI, ChromaDB, Streamlit

使用RAG技术构建一个能够回答客户问题的智能客服系统

---

## 🎯 案例概述

本案例展示如何使用RAG技术构建一个基础的智能客服系统，能够：
- 回答常见问题（FAQ）
- 支持多轮对话
- 提供友好的Web界面

### 技术亮点

- ✅ 基于RAG的问答系统
- ✅ OpenAI Embeddings语义检索
- ✅ ChromaDB向量数据库
- ✅ 多轮对话（对话历史管理）
- ✅ Streamlit Web界面

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd projects/case1-customer-service
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your-actual-api-key
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
TOP_K=3
```

### 3. 运行系统

```bash
streamlit run main.py
```

访问：http://localhost:8501

---

## 📁 项目结构

```
case1-customer-service/
├── main.py              # Streamlit主程序
├── rag_system.py        # RAG系统核心实现
├── knowledge_base.py    # 知识库管理
├── requirements.txt     # 依赖包
└── README.md           # 详细文档
```

---

## 💬 核心功能

### 1. FAQ问答
- 产品使用问题
- 配送政策
- 退换货流程
- 支付方式

### 2. 订单查询
- 订单状态查询
- 物流跟踪
- 配送时间

### 3. 产品推荐
- 基于需求推荐
- 产品对比
- 价格咨询

### 4. 多轮对话
- 上下文记忆
- 澄清问题
- 引导式查询

---

## 🔑 核心代码解析

### RAG系统初始化

```python
# rag_system.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

class RAGSystem:
    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings()

        # 加载知识库
        documents = self.load_knowledge_base()

        # 创建向量存储
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # 创建对话链
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
```

### 多轮对话处理

```python
def chat(message, history):
    """处理用户消息"""
    # 调用RAG系统
    response = qa_chain({
        "question": message,
        "chat_history": history
    })

    # 返回答案和来源
    return response["answer"], response["source_documents"]
```

---

## 📊 示例对话

**用户**: 退换货政策是什么？

**客服**: 根据知识库，退换货政策：支持7天无理由退换货，商品需保持完好，不影响二次销售。

**用户**: 运费谁承担？

**客服**: 退换货运费：因质量问题产生的退换货，运费由商家承担；因个人原因，运费由买家承担。

---

## 🎓 学习要点

通过本案例，你将学习：

1. **RAG基础架构**
   - 文档加载和分块
   - 向量嵌入和存储
   - 语义检索

2. **多轮对话实现**
   - 对话历史管理
   - 上下文维护
   - 提示词工程

3. **Web界面开发**
   - Streamlit基础
   - 会话状态管理
   - 用户界面设计

---

## 📈 扩展方向

### 短期优化
- [ ] 接入真实订单数据
- [ ] 集成更多知识源
- [ ] 添加用户反馈
- [ ] 优化对话管理

### 长期规划
- [ ] 接入真实客服系统
- [ ] 支持语音对话
- [ ] 多语言支持
- [ ] 生产环境部署

---

## 🛠️ 依赖版本

```txt
streamlit==1.29.0
langchain==0.1.0
chromadb==0.4.22
openai==1.7.2
python-dotenv==1.0.0
```

---

**[查看完整源码 →](https://github.com/vivy-yi/rag-tutorial/tree/main/projects/case1-customer-service)**

**[← 返回案例列表](index.md)**

---

**下一步**: 尝试[案例2：技术文档问答系统](case2-doc-qa.md)，学习混合检索！🚀
