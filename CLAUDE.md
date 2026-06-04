# CLAUDE.md

此文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指导。

## 项目概述

RAG（检索增强生成）知识库系统。支持 PDF/Markdown/TXT/DOCX 文档上传、向量检索和 LLM 生成回答。

## 架构

**核心引擎**（`rag_knowledge_base.py`）：五个类组成流水线：
- `Config` — 集中配置（分块大小、模型名称、提供商）
- `DocumentProcessor` — 加载 PDF/MD/TXT/DOCX，文本分块
- `EmbeddingManager` — 封装 sentence-transformers 用于本地嵌入
- `VectorStore` — ChromaDB 持久化存储（余弦相似度、HNSW 索引）
- `LLMGenerator` — 多提供商支持（OpenAI、Anthropic、Google、本地 Ollama）
- `RAGKnowledgeBase` — 编排器，串联所有组件

**Web 界面**（`app.py`）：Streamlit 实现的文档上传和问答界面。

## 目录结构

```
rag-knowledge-base/
├── rag_knowledge_base.py    # 核心 RAG 引擎（~550 行）
├── app.py                   # Streamlit Web 界面（~250 行）
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
├── data/                    # 数据存储目录（自动创建）
├── docs/                    # 文档和示例文档
├── vector_db/               # ChromaDB 持久化存储（自动创建）
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml   # 包含可选的 Qdrant 服务
    └── README.md
```

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt
pip install langchain-community beautifulsoup4  # 在线文档抓取需要

# 添加本地文档到知识库
python rag_knowledge_base.py --command add --file path/to/doc.pdf
python rag_knowledge_base.py --command add --dir docs --pattern "*.md"

# 添加在线文档到知识库
python rag_knowledge_base.py --command add --url https://example.com/doc

# 查询知识库
python rag_knowledge_base.py --command query --question "你的问题"

# 查看统计信息
python rag_knowledge_base.py --command stats

# 导出知识库
python rag_knowledge_base.py --command export --output export.json

# 启动 Web 界面
streamlit run app.py

# Docker 部署
cd docker && docker-compose up -d
```

## 配置说明

编辑 `config.yml` 文件即可修改配置（无需修改代码），或删除 `config.yml` 使用默认值：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_model` | BAAI/bge-large-zh-v1.5 | 本地嵌入模型 |
| `llm_provider` | openai | openai / anthropic / google / local |
| `llm_model` | gpt-4o | LLM 模型名称 |
| `vector_db` | chroma | chroma / faiss / qdrant |
| `chunk_size` | 500 | 每块最大 token 数 |
| `chunk_overlap` | 50 | 块之间重叠的 token 数 |
| `top_k` | 5 | 检索返回的文档块数量 |
| `similarity_threshold` | 0.7 | 相似度阈值 |

中文文档推荐将 `embedding_model` 改为 `paraphrase-multilingual-MiniLM-L12-v2` 或 `BAAI/bge-large-zh-v1.5`。

## API 密钥配置

API 密钥通过 `.env` 文件配置（**不要**放入 `config.yml`，避免提交到 git）：

```bash
# 1. 复制模板
cp .env.example .env

# 2. 编辑 .env 填入密钥
#    OPENAI_API_KEY=sk-xxx
#    ANTHROPIC_API_KEY=sk-ant-xxx
#    GOOGLE_API_KEY=xxx

# 3. 程序启动时自动加载 .env，无需额外操作
python rag_knowledge_base.py --command query --question "..."
```

### 对接国内模型（OpenAI 兼容接口）

国内模型（DeepSeek、通义千问、智谱等）大多兼容 OpenAI 接口格式，修改 `config.yml` 即可：

```yaml
llm_provider: "openai"
llm_model: "deepseek-chat"
llm_base_url: "https://api.deepseek.com/v1"
```

然后在 `.env` 中填入对应 API key。常见国内模型 endpoint：

| 服务 | base_url | 模型示例 |
|------|----------|---------|
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat |
| 阿里通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | glm-4 |
| 月之暗面 | `https://api.moonshot.cn/v1` | moonshot-v1 |
| 零一万物 | `https://api.lingyiwanwu.com/v1` | yi-lightning |

如果使用本地 Ollama（`llm_provider: local`），则无需 API 密钥。

## 关键设计说明

- `Config` 类使用类级别代码创建目录（`d.mkdir()` 在类定义时执行），新增路径时需注意此行为
- `EmbeddingManager` 采用懒加载方式加载模型（在首次调用 `embed_documents` 或 `embed_query` 时加载），而非初始化时加载
- `VectorStore` 的分块重叠实现直接将前一块的尾部拼接到当前块，对于中文内容可能在边界处产生乱码
- `app.py` Web 界面目前使用模拟数据（`st.session_state.searching` 路径），并未真正调用 `RAGKnowledgeBase` — 属于前端原型