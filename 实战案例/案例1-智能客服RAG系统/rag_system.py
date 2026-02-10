"""
案例1：智能客服RAG系统
RAG系统核心实现
"""

from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass

try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
except ImportError:
    print("警告：langchain未安装，将使用简化实现")
    # 使用简化实现
    OpenAIEmbeddings = None
    Chroma = None


@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    sources: List[str]
    confidence: float


class CustomerServiceRAG:
    """
    智能客服RAG系统

    功能：
    - 文档检索
    - 答案生成
    - 对话历史管理
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        初始化RAG系统

        Args:
            api_key: OpenAI API密钥
            model_name: LLM模型名称
            embedding_model: 嵌入模型名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            top_k: 检索文档数量
        """
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # 初始化组件
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 简化实现的文档存储
        self.documents = []

        # 检查API Key
        if not api_key:
            raise ValueError("请提供OPENAI_API_KEY")

    def add_documents(self, documents: List[str]) -> None:
        """
        添加文档到知识库

        Args:
            documents: 文档列表
        """
        try:
            if OpenAIEmbeddings is not None:
                # 使用langchain实现
                self._init_vector_store(documents)
            else:
                # 简化实现
                self.documents = documents

            print(f"✅ 已添加 {len(documents)} 个文档到知识库")
        except Exception as e:
            print(f"⚠️ 添加文档时出错: {str(e)}")
            # 降级到简化实现
            self.documents = documents

    def _init_vector_store(self, documents: List[str]) -> None:
        """初始化向量存储"""
        # 创建嵌入
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.embedding_model
        )

        # 创建向量存储
        self.vector_store = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        # 创建LLM
        llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.7
        )

        # 创建QA链
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        查询接口

        Args:
            question: 用户问题
            chat_history: 对话历史

        Returns:
            查询结果
        """
        try:
            if self.qa_chain is not None:
                # 使用langchain实现
                return self._query_with_langchain(question, chat_history)
            else:
                # 使用简化实现
                return self._query_simple(question)

        except Exception as e:
            print(f"⚠️ 查询时出错: {str(e)}")
            # 降级到简化实现
            return self._query_simple(question)

    def _query_with_langchain(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """使用langchain查询"""
        # 更新对话历史
        if chat_history:
            self.memory.clear()
            for msg in chat_history:
                if msg["role"] == "user":
                    self.memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.memory.add_ai_message(msg["content"])

        # 执行查询
        result = self.qa_chain({"question": question})

        # 提取来源
        sources = [
            doc.page_content
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "confidence": 0.85
        }

    def _query_simple(self, question: str) -> Dict[str, Any]:
        """简化实现：关键词匹配"""
        question_lower = question.lower()

        # 简单的关键词匹配
        matched_docs = []
        for doc in self.documents:
            score = sum(1 for word in question_lower.split() if word in doc.lower())
            if score > 0:
                matched_docs.append((doc, score))

        # 排序
        matched_docs.sort(key=lambda x: x[1], reverse=True)

        # 生成答案
        if matched_docs:
            top_doc = matched_docs[0][0]
            answer = f"根据知识库，{top_doc[:200]}..."
            sources = [doc for doc, _ in matched_docs[:3]]
            confidence = min(0.9, matched_docs[0][1] * 0.1)
        else:
            answer = "抱歉，我没有找到相关信息。您可以尝试重新表述问题，或联系人工客服。"
            sources = []
            confidence = 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    def clear_history(self) -> None:
        """清除对话历史"""
        if self.memory:
            self.memory.clear()


# 测试代码
if __name__ == "__main__":
    # 测试RAG系统
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("请设置OPENAI_API_KEY环境变量")
    else:
        # 创建系统
        rag = CustomerServiceRAG(api_key=api_key)

        # 添加测试文档
        test_docs = [
            "我们支持7天无理由退换货",
            "配送时间一般为2-3个工作日",
            "支持支付宝、微信支付和银行卡支付"
        ]
        rag.add_documents(test_docs)

        # 测试查询
        result = rag.query("退换货政策是什么？")
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence']}")
