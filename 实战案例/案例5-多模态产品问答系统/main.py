"""
案例5：多模态产品问答系统
主程序
"""

import streamlit as st
from multimodal_rag import MultimodalRAG
from PIL import Image
import io

st.set_page_config(page_title="多模态问答", page_icon="🖼️", layout="wide")

st.title("🖼️ 多模态产品问答系统")
st.markdown("### 支持图文混合查询的产品问答系统")

# 初始化
if "mm_rag" not in st.session_state:
    st.session_state.mm_rag = MultimodalRAG()

system = st.session_state.mm_rag

# 两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 上传产品图片")
    uploaded_file = st.file_uploader("上传产品图片", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        st.session_state.image = image

with col2:
    st.markdown("### 💬 文本描述")
    text_query = st.text_area("产品描述或问题", placeholder="例如：这个产品有什么特点？")

    mode = st.radio(
        "查询模式",
        ["图文结合", "仅图像", "仅文本"]
    )

# 查询按钮
if st.button("🔍 查询", type="primary"):
    if mode == "仅图像" and "image" not in st.session_state:
        st.warning("请先上传图片")
    elif mode == "仅文本" and not text_query:
        st.warning("请输入文本")
    else:
        with st.spinner("正在分析..."):
            # 准备输入
            image_data = st.session_state.get("image")
            query_text = text_query if mode != "仅图像" else ""

            # 执行查询
            result = system.query(
                image=image_data,
                text=query_text,
                mode=mode
            )

            # 显示结果
            st.markdown("---")
            st.markdown("### 📖 查询结果")
            st.write(result["answer"])

            if result.get("products"):
                st.markdown("### 🛍️ 推荐产品")
                for product in result["products"]:
                    st.markdown(f"- **{product['name']}**: {product['description']}")

            if result.get("confidence"):
                st.caption(f"置信度: {result['confidence']:.1%}")
