"""
RAG知识库 - Streamlit Web界面
快速启动: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime

from rag_knowledge_base import RAGKnowledgeBase, Config, logger


# ==================== 初始化 RAG 引擎 ====================

@st.cache_resource
def init_kb():
    """初始化知识库（只执行一次）"""
    kb = RAGKnowledgeBase()
    kb.initialize()
    return kb

kb = init_kb()


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

    try:
        kb_stats = kb.get_stats()
        doc_count = kb_stats.get("document_count", 0)
        st.metric("文档块数量", doc_count)
        st.metric("嵌入模型", kb_stats.get("embedding_model", ""))
        st.metric("LLM模型", f"{kb_stats.get('llm_provider')} / {kb_stats.get('llm_model')}")
    except Exception:
        st.metric("文档块数量", "0")
        st.metric("嵌入模型", Config.EMBEDDING_MODEL)
        st.metric("LLM模型", f"{Config.LLM_PROVIDER} / {Config.LLM_MODEL}")

    st.markdown("---")

    # 操作选项
    st.subheader("操作")

    if st.button("📤 添加文档", use_container_width=True):
        st.session_state.show_upload = True

    if st.button("🗑️ 清空知识库", use_container_width=True):
        st.session_state.show_upload = False
        st.info("清空功能暂未实现")

    st.markdown("---")

    # 配置
    with st.expander("⚙️ 配置", expanded=False):
        # API 密钥（动态设置，无需重启）
        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-xxx",
            help="设置后立即生效，无需重启"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        base_url = st.text_input(
            "API Base URL（可选）",
            value=Config.LLM_BASE_URL or "",
            placeholder="https://api.deepseek.com/v1",
            help="对接国内模型时填入对应的 base_url"
        )
        if base_url:
            Config.LLM_BASE_URL = base_url

        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索数量", 1, 10, Config.TOP_K)
        with col2:
            chunk_size = st.slider("分块大小", 200, 1000, Config.CHUNK_SIZE, step=50)

        # 日志级别
        log_level = st.selectbox(
            "日志级别",
            options=["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0,
            help="设置日志输出详细程度，DEBUG 显示最详细的信息"
        )
        if log_level != os.environ.get("LOG_LEVEL", "INFO"):
            os.environ["LOG_LEVEL"] = log_level
            logger.setLevel(getattr(logging, log_level, logging.INFO))

        if st.button("🔄 重新加载配置", use_container_width=True):
            Config.load(force=True)
            st.success("✅ 配置已重载")
            st.rerun()


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

            if st.button("开始上传并处理", use_container_width=True, key="process_upload"):
                success_count = 0
                fail_count = 0
                total_chunks = 0
                progress_bar = st.progress(0)

                for i, f in enumerate(uploaded_files):
                    try:
                        # 保存到临时文件
                        suffix = Path(f.name).suffix or ".tmp"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name

                        # 处理文档
                        chunks = kb.add_document(tmp_path)
                        total_chunks += chunks
                        success_count += 1

                        # 清理临时文件
                        os.unlink(tmp_path)

                    except Exception as e:
                        fail_count += 1
                        st.error(f"❌ {f.name} 处理失败: {e}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                if success_count > 0:
                    st.success(f"🎉 {success_count} 个文档处理完成，共 {total_chunks} 个文档块")
                if fail_count > 0:
                    st.warning(f"⚠️ {fail_count} 个文档处理失败")

                st.session_state.show_upload = False
                st.rerun()

    st.divider()

# 查询区域
st.subheader("🔍 查询知识库")

# 预设问题
preset_questions = [
    "RAG的工作原理是什么？",
    "如何优化RAG的检索效果？",
    "支持哪些文档格式？",
    "如何配置API密钥？",
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
    if st.button("🔄 清空", use_container_width=True):
        st.session_state.query = ""
        st.session_state.search_result = None
        st.rerun()

# 执行查询
if search_btn and query.strip():
    with st.spinner("正在检索知识库..."):
        try:
            result = kb.query(query, top_k=top_k)
            st.session_state.search_result = result
        except Exception as e:
            st.error(f"查询失败: {e}")
            st.session_state.search_result = None

# 显示查询结果
result = st.session_state.get("search_result")
if result:
    contexts = result.get("contexts", [])
    answer = result.get("answer", "")

    if contexts:
        st.subheader("📖 检索到的相关文档")

        for i, ctx in enumerate(contexts):
            meta = ctx.get("metadata", {})
            source = meta.get("filename", meta.get("source", "unknown"))
            similarity = ctx.get("similarity", 0)
            content = ctx.get("content", "")

            with st.container():
                st.markdown(f"""
                <div class="context-box">
                    <span class="source-tag">{source}</span>
                    <span style="color: #4caf50; font-weight: bold;">相似度: {similarity:.2f}</span>
                    <p style="margin-top: 0.5rem;">{content[:300]}{'...' if len(content) > 300 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("未检索到相关文档，请先上传文档到知识库")

    if answer:
        st.divider()
        st.subheader("💡 AI回答")
        st.markdown(f"""
        <div class="answer-box">
            {answer}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📌 回答基于以上检索到的文档生成 | {result.get('timestamp', '')}")


# ==================== 页脚 ====================

st.divider()
st.caption(f"📚 RAG知识库 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 基于LangChain + ChromaDB")