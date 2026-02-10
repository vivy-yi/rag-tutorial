"""
案例4：企业知识图谱问答系统
主程序
"""

import streamlit as st
from graph_rag import GraphRAGSystem
from knowledge_graph import build_sample_graph

st.set_page_config(page_title="知识图谱问答", page_icon="🔗", layout="wide")

st.title("🔗 企业知识图谱问答系统")
st.markdown("### 基于知识图谱的多跳推理问答")

# 初始化
if "graph_rag" not in st.session_state:
    with st.spinner("正在构建知识图谱..."):
        G = build_sample_graph()
        st.session_state.graph_rag = GraphRAGSystem(G)

system = st.session_state.graph_rag

# 查询界面
query = st.text_input("💬 请输入问题", placeholder="例如：张三负责哪些项目？")

if query:
    result = system.query(query)

    st.markdown("---")
    st.markdown("### 📖 答案")
    st.write(result["answer"])

    if result["reasoning_path"]:
        with st.expander("🔍 推理路径"):
            for step, item in enumerate(result["reasoning_path"], 1):
                st.markdown(f"{step}. {item}")
