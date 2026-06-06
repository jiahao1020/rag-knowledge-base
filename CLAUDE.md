# CLAUDE.md

此文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指导。

每次更新代码后默认更新文档。

## 项目概述

RAG（检索增强生成）知识库系统。支持 PDF/Markdown/TXT/DOCX/Excel 文档上传、向量检索和 LLM 生成回答。Excel 导入时自动检测数据类型，结构化数据（SQL修复类）会合并为自然语言描述后入库，提升检索精度。

## 架构

**核心引擎**（`rag_knowledge_base.py`）：五个类组成流水线：
- `Config` — 集中配置（分块大小、模型名称、提供商）
- `DocumentProcessor` — 加载 PDF/MD/TXT/DOCX/Excel，文本分块；Excel 智能检测（`_detect_excel_type`）判断是否需要清洗
- `EmbeddingManager` — 封装 sentence-transformers 用于本地嵌入
- `VectorStore` — ChromaDB 持久化存储（余弦相似度、HNSW 索引）
- `LLMGenerator` — 多提供商支持（OpenAI、Anthropic、Google、本地 Ollama）
- `RAGKnowledgeBase` — 编排器，串联所有组件

**Web 界面**（`app.py`）：Streamlit 实现的文档上传和问答界面，已对接真实 RAG 引擎。通过 `@st.cache_resource` 缓存引擎实例，上传文件保存到临时路径后调用 `kb.add_document()` 处理，查询调用 `kb.query()` 获取真实检索结果和 LLM 回答。

## 目录结构

```
rag-knowledge-base/
├── rag_knowledge_base.py    # 核心 RAG 引擎（~550 行）
├── app.py                   # Streamlit Web 界面（~250 行）
├── config.yml               # 配置文件
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
├── .gitignore
├── .streamlit/
│   └── config.toml          # Streamlit 配置（上传大小限制等）
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
pip install cryptography  # 加密 PDF 文件需要

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
| `llm_base_url` | 无（使用官方 API） | 兼容 OpenAI 接口的自定义地址 |
| `vector_db` | chroma | chroma / faiss / qdrant |
| `chunk_size` | 500 | 每块最大 token 数 |
| `chunk_overlap` | 50 | 块之间重叠的 token 数 |
| `top_k` | 5 | 检索返回的文档块数量 |
| `similarity_threshold` | 0.7 | 相似度阈值 |

中文文档推荐将 `embedding_model` 改为 `paraphrase-multilingual-MiniLM-L12-v2` 或 `BAAI/bge-large-zh-v1.5`。

### 国内网络下载模型

首次运行需要下载嵌入模型（~1.3GB），如 HuggingFace 被墙，自动回退到 ModelScope 下载。也可手动执行：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5')"
streamlit run app.py
```

## API 密钥配置

API 密钥通过以下方式配置（**不要**放入 `config.yml`，避免提交到 git）：

### 方式一：Web 界面动态配置（无需重启）

启动 Web 界面后，在侧边栏 **⚙️ 配置** 中直接填入：
- **API Key** — 输入后立即生效
- **API Base URL** — 对接国内模型时填入对应的 base_url（如 `https://api.deepseek.com/v1`）

### 方式二：.env 文件

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

修改 `config.yml` 后，可在 Web 界面侧边栏点击 **🔄 重新加载配置** 按钮生效，无需重启。

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
| 商汤日日新 | `https://token.sensenova.cn/v1` | sensenova-6.7-flash-lite |
| 阿里通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | glm-4 |
| 月之暗面 | `https://api.moonshot.cn/v1` | moonshot-v1 |
| 零一万物 | `https://api.lingyiwanwu.com/v1` | yi-lightning |

如果使用本地 Ollama（`llm_provider: local`），则无需 API 密钥。

## 日志配置

通过 `LOG_LEVEL` 环境变量控制日志详细程度，默认 `INFO`：

```bash
# 仅显示 WARNING 及以上级别
LOG_LEVEL=WARNING python rag_knowledge_base.py --command stats

# 显示 DEBUG 详细信息（调试用）
LOG_LEVEL=DEBUG python rag_knowledge_base.py --command stats
```

日志输出格式：`[2026-06-06 09:48:54] [INFO] 消息`

日志同时输出到：
- **终端**（stderr）— 实时查看
- **文件** `logs/rag_kb.log` — 自动轮转，单个文件最大 5MB，保留 3 个备份

Web 界面侧边栏 ⚙️ 配置中也可直接切换日志级别，无需重启。

## 常见问题

### PDF 上传失败（提示 cryptography 错误）

部分 PDF 使用了 AES 加密，需要额外安装依赖：

```bash
pip install cryptography
```

### Web 端上传超过 200MB 限制

修改 `.streamlit/config.toml` 中的 `maxUploadSize`（单位 MB）：

```toml
[server]
maxUploadSize = 500  # 改为 500MB
```

修改后重启 Streamlit 生效。

### 启动慢或模型加载慢

嵌入模型采用**延迟加载**策略：启动时只初始化 ChromaDB（约 1-2 秒），模型在首次查询或上传文档时才载入内存，避免启动等待。

### 模型下载慢或连接失败

嵌入模型首次加载需下载 ~1.3GB。代码自动按以下顺序尝试：
1. 本地缓存
2. ModelScope（国内可访问）
3. HuggingFace（兜底）

也可手动下载：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5')"
streamlit run app.py
```

## 关键设计说明

- `Config` 类使用类级别代码创建目录（`d.mkdir()` 在类定义时执行），新增路径时需注意此行为
- `EmbeddingManager` 采用懒加载方式加载模型（在首次调用 `embed_documents` 或 `embed_query` 时加载），而非初始化时加载
- `VectorStore` 的分块重叠实现直接将前一块的尾部拼接到当前块，对于中文内容可能在边界处产生乱码