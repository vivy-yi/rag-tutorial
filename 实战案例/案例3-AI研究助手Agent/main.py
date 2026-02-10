"""
案例3：AI研究助手Agent
主程序 - Streamlit Web应用
"""

import streamlit as st
from dotenv import load_dotenv
import os

from research_agent import ResearchAgent
from tools import (
    SearchTool,
    ArxivTool,
    WikipediaTool,
    summarize_paper
)

# 页面配置
st.set_page_config(
    page_title="AI研究助手",
    page_icon="🔬",
    layout="wide"
)

load_dotenv()


def main():
    st.title("🔬 AI研究助手")
    st.markdown("### 自主规划、搜索、总结的AI研究助手")

    # 初始化Agent
    if "agent" not in st.session_state:
        st.session_state.agent = ResearchAgent(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    agent = st.session_state.agent

    # 研究目标输入
    research_topic = st.text_area(
        "📝 研究主题",
        placeholder="例如：研究Transformer在自然语言处理中的应用",
        height=100
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_papers = st.slider("论文数量", 3, 10, 5)
    with col2:
        include_arxiv = st.checkbox("包含ArXiv论文", value=True)
    with col3:
        include_wikipedia = st.checkbox("包含维基百科", value=True)

    if st.button("🚀 开始研究", type="primary"):
        if research_topic:
            with st.spinner(f"正在研究：{research_topic}"):
                try:
                    result = agent.research(
                        topic=research_topic,
                        max_papers=max_papers,
                        use_arxiv=include_arxiv,
                        use_wikipedia=include_wikipedia
                    )

                    # 显示结果
                    st.markdown("---")
                    st.markdown("## 📊 研究报告")
                    st.markdown(result["report"])

                    # 显示来源
                    if result["sources"]:
                        st.markdown("---")
                        st.markdown("## 📚 参考文献")

                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"文献 {i}: {source['title']}", expanded=i <= 2):
                                st.markdown(f"**作者**: {source.get('authors', 'N/A')}")
                                st.markdown(f"**年份**: {source.get('year', 'N/A')}")
                                st.markdown(f"**摘要**: {source.get('abstract', 'N/A')[:300]}...")
                                if source.get('url'):
                                    st.markdown(f"**链接**: [{source['url']}]({source['url']})")

                except Exception as e:
                    st.error(f"❌ 研究失败: {str(e)}")
        else:
            st.warning("请输入研究主题")


if __name__ == "__main__":
    main()
