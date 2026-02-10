"""
案例1：智能客服RAG系统
主程序 - Streamlit Web应用
"""

import streamlit as st
from typing import List, Dict
import os
from dotenv import load_dotenv

from rag_system import CustomerServiceRAG
from knowledge_base import load_faq_documents

# 页面配置
st.set_page_config(
    page_title="智能客服系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载环境变量
load_dotenv()

# 初始化RAG系统
@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统（缓存）"""
    try:
        # 加载FAQ知识库
        documents = load_faq_documents()

        # 创建RAG系统
        rag_system = CustomerServiceRAG(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo"
        )

        # 添加文档
        rag_system.add_documents(documents)

        return rag_system
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        return None


def main():
    """主函数"""

    # 标题
    st.title("🤖 智能客服系统")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")

        # 系统信息
        st.info("""
        **本系统功能**：
        - 回答常见问题
        - 订单查询
        - 产品推荐
        - 多轮对话
        """)

        # 清除对话历史
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 初始化RAG系统
    rag_system = initialize_rag_system()

    if rag_system is None:
        st.error("⚠️ 系统初始化失败，请检查配置")
        return

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入
    if prompt := st.chat_input("💬 请输入您的问题..."):

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 添加到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 生成回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                try:
                    # 获取对话历史（最近5轮）
                    chat_history = st.session_state.messages[-10:-1] if len(st.session_state.messages) > 1 else []

                    # RAG查询
                    result = rag_system.query(
                        question=prompt,
                        chat_history=chat_history
                    )

                    # 显示答案
                    answer = result["answer"]
                    st.markdown(answer)

                    # 显示来源（如果有）
                    if result.get("sources") and len(result["sources"]) > 0:
                        with st.expander("📚 查看参考来源"):
                            for i, source in enumerate(result["sources"][:3], 1):
                                st.markdown(f"**来源{i}**: {source}")

                    # 添加到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"❌ 处理请求时出错: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，处理您的请求时出现了错误。"
                    })

    # 底部信息
    st.markdown("---")
    st.caption("💡 提示：您可以询问关于产品、订单、配送、退换货等问题")


if __name__ == "__main__":
    main()
