"""
RAG知识库 - Streamlit Web界面
快速启动: streamlit run app.py
"""

import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="RAG知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式 ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .context-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .source-tag {
        display: inline-block;
        background: #2196f3;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏 ====================

with st.sidebar:
    st.header("📚 RAG知识库")
    
    st.markdown("---")
    
    # 知识库状态
    st.subheader("知识库状态")
    
    # 模拟统计（实际应从后端获取）
    stats = {
        "文档数量": "12",
        "文档块数量": "156",
        "嵌入模型": "all-MiniLM-L6-v2",
        "最后更新": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    for key, value in stats.items():
        st.metric(key, value)
    
    st.markdown("---")
    
    # 操作选项
    st.subheader("操作")
    
    if st.button("📤 添加文档", use_container_width=True):
        st.session_state.show_upload = True
    
    if st.button("🗑️ 清空知识库", use_container_width=True):
        st.warning("确定要清空知识库吗？")
    
    st.markdown("---")
    
    # 配置
    with st.expander("⚙️ 配置"):
        st.selectbox("嵌入模型", [
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "bge-large-zh-v1.5"
        ])
        st.selectbox("LLM模型", [
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet",
            "gemini-1.5-pro"
        ])
        st.slider("检索数量", 1, 10, 5)
        st.slider("分块大小", 200, 1000, 500)


# ==================== 主页面 ====================

st.markdown('<div class="main-header">📚 RAG 知识库</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于检索增强生成的智能问答系统</div>', unsafe_allow_html=True)

# 上传文档区域
if st.session_state.get("show_upload", False):
    with st.expander("📤 上传文档", expanded=True):
        uploaded_files = st.file_uploader(
            "选择要上传的文档",
            type=["pdf", "md", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")
            
            if st.button("开始上传并处理", use_container_width=True):
                with st.spinner("正在处理文档..."):
                    # 模拟处理
                    for f in uploaded_files:
                        st.success(f"✅ {f.name}")
                
                st.success("🎉 文档处理完成！")
                st.session_state.show_upload = False
    
    st.divider()

# 查询区域
st.subheader("🔍 查询知识库")

# 预设问题
preset_questions = [
    "如何设置API密钥？",
    "RAG的工作原理是什么？",
    "如何部署到生产环境？",
    "支持哪些文档格式？",
]

col1, col2, col3, col4 = st.columns(4)
for i, q in enumerate(preset_questions):
    with [col1, col2, col3, col4][i % 4]:
        if st.button(q, use_container_width=True, key=f"preset_{i}"):
            st.session_state.query = q

st.divider()

# 查询输入
query = st.text_area(
    "输入你的问题",
    placeholder="例如：如何优化RAG的检索效果？",
    height=100,
    key="query"
)

col_search, col_clear = st.columns([3, 1])
with col_search:
    search_btn = st.button("🔍 搜索", type="primary", use_container_width=True)
with col_clear:
    st.button("🔄 清空", use_container_width=True)

# 显示搜索结果
if search_btn and query:
    st.session_state.searching = True

if st.session_state.get("searching", False) and query:
    with st.spinner("正在检索知识库..."):
        # 模拟检索结果
        contexts = [
            {
                "content": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。核心原理是：先从知识库检索相关文档，然后将检索结果作为上下文输入LLM生成回答。",
                "source": "docs/README.md",
                "similarity": 0.92
            },
            {
                "content": "要优化RAG检索效果，可以：1. 调整分块大小（200-1000 tokens）2. 使用混合检索（向量+关键词）3. 添加Rerank重排序 4. 优化嵌入模型",
                "source": "docs/README.md",
                "similarity": 0.87
            },
            {
                "content": "支持PDF、Markdown、TXT、DOCX等多种格式。文档上传后会自动分块、向量化并存储到向量数据库中。",
                "source": "docs/README.md",
                "similarity": 0.75
            }
        ]
        
        # 模拟LLM回答
        answer = """根据检索到的信息，优化RAG检索效果的方法包括：

**1. 调整分块策略**
- 分块大小建议在200-1000 tokens之间
- 适当增加块重叠（50-100 tokens）可以提高检索连续性

**2. 使用混合检索**
- 结合向量检索（语义相似）和关键词检索（BM25）
- 权重建议：向量0.7 + 关键词0.3

**3. 添加Rerank重排序**
- 对初步检索结果使用Cross-Encoder重新排序
- 可以显著提升Top-K的准确性

**4. 优化嵌入模型**
- 中文场景推荐使用：bge-large-zh-v1.5
- 多语言场景：paraphrase-multilingual-MiniLM-L12-v2

**5. 文档质量优化**
- 清理格式、去除冗余内容
- 添加结构化标题和元数据"""

    # 显示上下文
    st.subheader("📖 检索到的相关文档")
    
    for i, ctx in enumerate(contexts):
        with st.container():
            st.markdown(f"""
            <div class="context-box">
                <span class="source-tag">{ctx['source']}</span>
                <span style="color: #4caf50; font-weight: bold;">相似度: {ctx['similarity']:.2f}</span>
                <p style="margin-top: 0.5rem;">{ctx['content'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # 显示回答
    st.subheader("💡 AI回答")
    st.markdown(f"""
    <div class="answer-box">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # 引用来源
    st.caption("📌 回答基于以上检索到的文档生成")
    
    st.session_state.searching = False


# ==================== 页脚 ====================

st.divider()
st.caption(f"📚 RAG知识库 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 基于LangChain + ChromaDB")
