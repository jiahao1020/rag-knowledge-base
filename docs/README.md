# RAG知识库系统 - 完整文档

## 📋 目录

- [项目概述](#-项目概述)
- [实现功能清单](#-实现功能清单)
- [技术架构详解](#-技术架构详解)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [进阶功能](#-进阶功能)
- [常见问题](#-常见问题)

---

## 📖 项目概述

### 什么是RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合**检索**和**生成**的AI技术。

### 核心原理

1. **检索（Retrieval）**：从知识库中检索与问题相关的文档片段
2. **增强（Augmented）**：将检索到的信息作为上下文
3. **生成（Generation）**：LLM基于上下文生成准确回答

### 优势

- ✅ **减少幻觉**：基于真实文档回答，减少胡说八道
- ✅ **可追溯**：每个回答都有信息来源
- ✅ **知识更新**：只需更新文档，无需重新训练模型
- ✅ **成本低**：无需微调大模型

### 应用场景

| 场景 | 说明 |
|------|------|
| **企业知识库** | 员工手册查询、产品文档检索、技术支持问答 |
| **个人知识管理** | 笔记检索（Obsidian/Notion）、会议记录查询 |
| **客户服务** | 智能客服机器人、产品FAQ自动回答 |
| **采购软件集成** | 客户自助查询、工单自动分类 |

---

## ✅ 实现功能清单

### 一、核心功能（`rag_knowledge_base.py`）

| 功能模块 | 实现内容 | 代码行数 |
|---------|---------|---------|
| **文档加载** | 支持PDF、Markdown、TXT、DOCX、Excel五种格式 | ~95行 |
| **文本分块** | 递归分块（tiktoken 精确计数，Markdown 标题强制切分，句子/字符降级） | ~110行 |
| **向量化** | 本地嵌入模型（sentence-transformers） | ~40行 |
| **向量存储** | ChromaDB持久化存储，支持查询和统计 | ~80行 |
| **LLM生成** | 支持OpenAI、Anthropic、Google、本地Ollama四种提供商 | ~120行 |
| **RAG检索** | 向量检索 + 上下文构建 + 答案生成 | ~60行 |
| **CLI接口** | 命令行工具，支持add/query/stats/export命令 | ~100行 |

**核心代码总计**：~540行

### 二、Web界面（`app.py`）

| 功能模块 | 实现内容 |
|---------|---------|
| **页面布局** | 侧边栏 + 主内容区，响应式设计 |
| **知识库状态** | 显示文档数量、嵌入模型、最后更新等 |
| **文档上传** | 多文件拖拽上传，支持PDF/MD/TXT/DOCX |
| **预设问题** | 5个预设问题快捷查询 |
| **查询输入** | 大文本框输入问题 |
| **结果展示** | 检索到的文档块 + 相似度 + AI回答 |
| **样式美化** | 自定义CSS，卡片式布局 |

### 三、部署方案（`docker/`）

| 文件 | 内容 |
|------|------|
| `Dockerfile` | Python 3.11基础镜像，预装所有依赖 |
| `docker-compose.yml` | 一键启动，包含Qdrant向量数据库选项 |
| `README.md` | Docker部署完整指南 |

### 四、代码统计

| 文件 | 行数 | 说明 |
|------|------|------|
| `rag_knowledge_base.py` | ~850 | 核心引擎（含Excel智能清洗、tiktoken 分块、清空功能） |
| `app.py` | ~300 | Web界面（响应式移动端适配、清空按钮） |
| `requirements.txt` | ~20 | 依赖列表 |
| `.streamlit/config.toml` | ~3 | Streamlit配置（上传大小限制） |
| `docker/Dockerfile` | ~25 | Docker镜像 |
| `docker/docker-compose.yml` | ~35 | 一键部署 |
| `docs/README.md` | ~200 | 使用文档 |

---

## 🏗️ 技术架构详解

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 知识库系统架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      应用层 (Application)                     │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │   │
│  │  │ CLI命令行   │    │ Web界面     │    │ Python API  │       │   │
│  │  │ rag_knowledge│   │ Streamlit   │    │ (可集成)    │       │   │
│  │  │ _base.py    │    │ app.py      │    │             │       │   │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘       │   │
│  └─────────┼──────────────────┼──────────────────┼──────────────┘   │
│            │                  │                  │                  │
│  ┌─────────▼──────────────────▼──────────────────▼──────────────┐   │
│  │                      核心引擎层 (Core Engine)                 │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │   │
│  │  │ 文档处理器  │───▶│  文本分块   │───▶│  向量化     │       │   │
│  │  │ Document    │    │ Chunking    │    │ Embedding   │       │   │
│  │  │ Processor   │    │             │    │ Manager     │       │   │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘       │   │
│  │         │                  │                  │               │   │
│  │         │                  │                  ▼               │   │
│  │         │                  │          ┌─────────────┐         │   │
│  │         │                  │          │ 向量数据库  │         │   │
│  │         │                  │          │ VectorStore │         │   │
│  │         │                  │          │ (ChromaDB)  │         │   │
│  │         │                  │          └──────┬──────┘         │   │
│  │         │                  │                 │                │   │
│  │         │                  │                 ▼                │   │
│  │         │                  │          ┌─────────────┐         │   │
│  │         │                  │          │  向量检索   │         │   │
│  │         │                  │          │  Search     │         │   │
│  │         │                  │          └──────┬──────┘         │   │
│  │         │                  │                 │                │   │
│  │         │                  │                 ▼                │   │
│  │         │                  │          ┌─────────────┐         │   │
│  │         │                  │          │ 上下文构建  │         │   │
│  │         │                  │          │ Context     │         │   │
│  │         │                  │          │ Building    │         │   │
│  │         │                  │          └──────┬──────┘         │   │
│  │         │                  │                 │                │   │
│  │         │                  │                 ▼                │   │
│  │         │                  │          ┌─────────────┐         │   │
│  │         │                  │          │   LLM生成   │         │   │
│  │         │                  │          │   Generator │         │   │
│  │         │                  │          └──────┬──────┘         │   │
│  │         │                  │                 │                │   │
│  └─────────┼──────────────────┼─────────────────┼────────────────┘   │
│            │                  │                 │                    │
│  ┌─────────▼──────────────────▼─────────────────▼──────────────┐   │
│  │                      数据层 (Data Layer)                     │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │   │
│  │  │ docs/       │    │ data/       │    │ vector_db/  │       │   │
│  │  │ 原始文档    │    │ 导出数据    │    │ ChromaDB    │       │   │
│  │  │ (PDF/MD)    │    │ (JSON)      │    │ 向量存储    │       │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 技术栈详解

| 层级 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **编程语言** | Python | 3.10+ | 核心开发语言 |
| **嵌入模型** | sentence-transformers | BAAI/bge-large-zh-v1.5 | 本地运行，免费 |
| **向量数据库** | ChromaDB | 0.4+ | 轻量级，无需服务 |
| **LLM框架** | LangChain | 0.1+ | 主流框架，生态丰富 |
| **文档处理** | pypdf, python-docx | 最新 | PDF/Word解析 |
| **Web界面** | Streamlit | 1.28+ | 快速构建，无需前端 |
| **容器化** | Docker | 最新 | 一键部署 |

### 数据流程

#### 文档入库流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      文档入库流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户上传文档 ──▶ 格式检测 ──▶ 文本提取 ──▶ 文本分块            │
│                                              │                  │
│                                              ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  分块策略 (递归分块):                                     │   │
│  │  - 默认块大小: 512 tokens (tiktoken 精确计数)             │   │
│  │  - 块重叠: 80 tokens (~15%, 保证上下文连续性)             │   │
│  │  - 切分优先级: Markdown 标题 → 段落 → 句子 → 字符        │   │
│  │  - Markdown 标题强制切分，保证标题和内容不分离            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                              │                  │
│                                              ▼                  │
│  向量化 ──▶ 生成向量(384维) ──▶ 添加元数据 ──▶ 存入ChromaDB     │
│                                                                 │
│  元数据包含: source, filename, chunk_index, collection, time    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 查询回答流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      查询回答流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户提问 ──▶ 向量化 ──▶ 向量检索 ──▶ 获取Top-K相关文档块       │
│                                              │                  │
│                                              ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  检索策略:                                                │   │
│  │  - 相似度算法: Cosine Similarity                          │   │
│  │  - 默认返回: Top 5 个最相关文档块                         │   │
│  │  - 相似度阈值: 0.7 (可调整)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                              │                  │
│                                              ▼                  │
│  构建Prompt ──▶ 包含检索到的上下文 ──▶ 发送给LLM                │
│                                              │                  │
│                                              ▼                  │
│  LLM生成回答 ──▶ 返回带引用的答案 ──▶ 展示给用户                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心代码结构

```python
# rag_knowledge_base.py 类结构

class Config:
    """系统配置 - 所有可调参数集中管理"""
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o"
    CHUNK_SIZE = 512      # token 数，tiktoken 精确计数
    CHUNK_OVERLAP = 80    # token 数，~15%
    TOP_K = 5

class DocumentProcessor:
    """文档处理 - 多格式支持"""
    load_document()          # 加载PDF/MD/TXT/DOCX/Excel
    split_text()             # 递归分块（tiktoken 精确计数，Markdown 标题强制切分）
    _load_excel_as_rows()    # Excel逐行读取
    _detect_excel_type()     # 自动检测是否需要清洗
    _add_excel_rows_raw()    # 普通表格逐行存储
    _add_excel_rows_cleaned() # 结构化数据合并为自然语言

class EmbeddingManager:
    """向量化 - 本地嵌入模型"""
    load_model()         # 加载sentence-transformers
    embed_documents()    # 批量生成向量
    embed_query()        # 查询向量化

class VectorStore:
    """向量数据库 - ChromaDB封装"""
    initialize()         # 初始化数据库
    add_documents()      # 添加文档块
    clear()              # 清空集合
    search()             # 向量检索
    get_stats()          # 获取统计信息

class LLMGenerator:
    """LLM生成 - 多提供商支持"""
    generate_answer()    # 生成回答
    _call_openai()       # OpenAI API
    _call_anthropic()    # Claude API
    _call_google()       # Gemini API
    _call_local()        # Ollama本地模型

class RAGKnowledgeBase:
    """主类 - 整合所有组件"""
    initialize()         # 初始化
    add_document()       # 添加单文档
    add_directory()      # 添加目录
    query()              # 查询
    clear()              # 清空知识库
    get_stats()          # 统计
    export()             # 导出
```

### 支持的LLM提供商

| 提供商 | 模型示例 | 成本 | 代码实现 |
|--------|---------|------|---------|
| **OpenAI** | gpt-4o, gpt-3.5-turbo | 付费 | `_call_openai()` |
| **Anthropic** | claude-3-5-sonnet | 付费 | `_call_anthropic()` |
| **Google** | gemini-1.5-pro | 付费 | `_call_google()` |
| **本地** | llama3, qwen | 免费 | `_call_local()` |

### 扩展性设计

| 扩展点 | 当前实现 | 可扩展方向 |
|--------|---------|-----------|
| **文档格式** | PDF/MD/TXT/DOCX/Excel | 添加HTML、PPT支持 |
| **Excel智能导入** | 自动检测+行级存储/结构化清洗 | 支持多Sheet、按任务类别分组 |
| **向量数据库** | ChromaDB | 支持Qdrant、Pinecone、Milvus |
| **检索策略** | 向量检索 | 添加混合检索（BM25+向量） |
| **重排序** | 无 | 添加Cross-Encoder Rerank |
| **多集合** | 支持 | 按业务分类管理 |
| **API接口** | CLI | 添加REST API |

---

## 🚀 快速开始

### 方案A：本地运行（需要pip环境）

```bash
# 1. 安装依赖
cd rag-knowledge-base
pip install -r requirements.txt

# 2. 配置
cp .env.example .env          # 创建环境变量文件
# 编辑 .env 填入你的API密钥
# 可选：编辑 config.yml 修改模型、分块等参数

# 3. 添加本地文档
python rag_knowledge_base.py --command add --dir docs --pattern "*.md"

# 4. 添加在线文档
python rag_knowledge_base.py --command add --url https://example.com/doc

# 5. 查询
python rag_knowledge_base.py --command query --question "如何设置API密钥？"

# 6. 启动Web界面
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 方案B：Docker部署（推荐）

```bash
# 1. 配置环境变量
cp docker/.env.example docker/.env
# 编辑 docker/.env 填入API密钥

# 2. 构建并运行
cd docker
docker-compose up -d

# 3. 访问
浏览器打开: http://localhost:8501
```

### 方案C：云端无代码平台（最快）

直接去以下平台，上传文档就能用，无需部署：

| 平台 | 免费额度 | 链接 |
|------|---------|------|
| **Dify** | 免费试用 | [dify.ai](https://dify.ai) |
| **Flowise** | 免费自托管 | [flowiseai.com](https://flowiseai.com) |

---

## ⚙️ 配置说明

编辑 `config.yml` 文件即可修改配置（无需修改代码），或删除 `config.yml` 使用默认值：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_model` | BAAI/bge-large-zh-v1.5 | 嵌入模型（本地） |
| `llm_provider` | openai | LLM提供商 |
| `llm_model` | gpt-4o | LLM模型 |
| `llm_base_url` | 无（使用官方API） | 兼容OpenAI接口的自定义地址 |
| `chunk_size` | 512 | 分块大小（token，tiktoken 精确计数） |
| `chunk_overlap` | 80 | 块重叠（token，~15%） |
| `top_k` | 5 | 检索返回数量 |
| `similarity_threshold` | 0.7 | 相似度阈值 |

### API密钥配置

API密钥支持以下方式配置：

**方式一：Web 界面动态配置（推荐）**
启动 Web 界面后，在侧边栏 ⚙️ 配置中直接填入 API Key 和 Base URL，立即生效，无需重启。

**方式二：.env 文件**
复制 `.env.example` 为 `.env` 并填入密钥，程序启动时自动加载。修改 `config.yml` 后可在 Web 界面点击"重新加载配置"按钮生效。

### LLM提供商切换

```python
# OpenAI
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"

# Anthropic (Claude)
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-3-5-sonnet"

# Google (Gemini)
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-1.5-pro"

# 本地模型 (Ollama)
LLM_PROVIDER = "local"
LLM_MODEL = "llama3"
```

### 对接国内模型

国内模型（DeepSeek、通义千问、智谱等）大多兼容 OpenAI 接口格式，修改 `config.yml`：

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

### 日志级别控制

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

### 中文优化

```python
# 使用中文优化的嵌入模型
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 或
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
```

---

## 📈 进阶功能

### 1. 文本分块策略

分块质量直接影响 RAG 检索效果。当前采用**递归分块**策略：

**切分优先级**：Markdown 标题 → 段落 → 句子 → 字符

- **Markdown 标题强制切分**：遇到 `#` 到 `######` 标题时立即切分，保证标题和内容在同一块
- **tiktoken 精确计数**：使用 `cl100k_base` 编码精确计算 token 数，而非字符数
- **句子降级拆分**：段落超过 chunk_size 时，按中英文标点（。！？.!?）拆分为句子
- **字符级兜底**：单个句子仍超长时，逐字符拆分确保不超过阈值

**默认参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 512 | token 数（约 768 中文字符） |
| `chunk_overlap` | 80 | ~15%，保证跨块上下文连续性 |

**为什么重要**：
- 标题和内容分离 → 检索时只能命中标题或内容，LLM 看不到完整上下文
- 字符数计数 → 中文 token 数约为字符数的 1.5 倍，容易超出预期
- 重叠太小 → 跨块语义信息丢失

```python
# 示例：订单状态修复文档
# 之前（旧算法）：标题和修复脚本在不同块 → 检索失败
# 现在（新算法）：标题强制切分，标题+查询+修复脚本在同一块 → 检索成功
```

### 2. 多集合管理

```python
kb.add_document("file.pdf", collection_name="采购合同")
kb.add_document("file.md", collection_name="产品文档")

# 查询时指定集合
results = kb.vector_store.search(query_embedding, top_k=5, filter={"collection": "采购合同"})
```

### 3. Excel 智能导入

导入 Excel 时自动检测数据类型，决定清洗策略：

**自动检测规则：**
- 列名含 `任务类别`、`查询脚本`、`SQL`、`修复` 等关键词 → 结构化清洗
- 单元格含 `update`、`insert into`、`delete from` 等 SQL 关键词 → 结构化清洗
- 普通表格（商品名/价格/库存等）→ 逐行存储

**结构化清洗效果（以修复类 Excel 为例）：**

```python
# 原始 Excel 行数据（碎片化）
Row 5:  任务类别=订单类型 | 前置查询脚本=select... | 修复项=修复字段 | 订单头表=po_type_id
Row 6:  修复项=修复脚本   | 订单头表=update sodr_po_header set po_type_id=?

# 清洗后入库（自然语言描述，查询和修复合并为一段）
任务类别：订单类型。
前置查询脚本：select order_type_id,order_type_code,order_type_name from sodr_order_type...
修复项：修复字段。修复字段（表：订单头表）：po_type_id。
修复项：修复脚本。修复脚本（表：订单头表）：update sodr_po_header set po_type_id=...
涉及表：订单头表。
```

**多 Sheet 支持：** 自动遍历所有 sheet，按 sheet 分组处理

**支持格式：** `.xlsx` / `.xls`

```python
from langchain.retrievers import BM25Retriever, VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_texts(texts)
vector_retriever = VectorStoreRetriever(vectorstore=vectorstore)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

### 4. Rerank重排序

```python
from langchain_community.cross_encoders import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# 对检索结果重新排序
```

### 5. REST API（可扩展）

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    result = kb.query(request.question, request.top_k)
    return result
```

---

## ❓ 常见问题

### Q: 为什么回答不准确？

A: 可能原因：
1. 文档质量差 → 清理文档格式
2. 分块不合理 → 调整 CHUNK_SIZE，确保标题和内容在同一块
3. 检索不到相关内容 → 检查嵌入模型
4. LLM理解偏差 → 优化prompt

### Q: 分块大小和重叠如何设置？

A: 默认 `chunk_size=512` token，`chunk_overlap=80` token（~15%）。
- **增大 chunk_size**：适合长文档，但可能包含无关信息
- **减小 chunk_size**：适合短文档，但可能丢失上下文
- **增大 overlap**：提高跨块召回率，但增加冗余
- **减小 overlap**：减少冗余，但可能丢失跨块语义

**推荐值**：
| 文档类型 | chunk_size | chunk_overlap |
|----------|-----------|---------------|
| 技术文档/Markdown | 512 | 80 |
| 长篇小说/论文 | 768 | 100 |
| 短问答/FAQ | 256 | 40 |

### Q: 如何支持中文？

A: 使用中文优化的嵌入模型：
```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 或
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
```

### Q: 本地部署如何保护隐私？

A: 使用本地模型：
```python
LLM_PROVIDER = "local"  # Ollama
LLM_MODEL = "llama3"
```

### Q: 启动慢怎么办？

A: 嵌入模型采用**延迟加载**策略，启动时只初始化 ChromaDB（约 1-2 秒），嵌入模型在首次查询或上传文档时才载入内存。第一次加载后后续重启会使用本地缓存，速度更快。

### Q: tiktoken 安装失败怎么办？

A: tiktoken 用于精确计数 token。如果安装失败，系统会自动回退到字符数估算（`len(s) / 1.5`），不影响功能，只是分块精度稍低。
```bash
pip install tiktoken
```
国内网络可尝试：
```bash
pip install tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 模型下载失败/被墙怎么办？

A: 内置 ModelScope 备选源，自动切换。也可手动下载：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5')"
```

### Q: PDF 上传报错 cryptography 怎么办？

A: 部分 PDF 用了 AES 加密，安装：
```bash
pip install cryptography
```

### Q: Web 上传超过 200MB 怎么办？

A: 编辑 `.streamlit/config.toml`，增大限制：
```toml
[server]
maxUploadSize = 1000  # 改为 1GB
```

### Q: 性能如何优化？

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 检索速度 | 使用HNSW索引 | 10-100倍提升 |
| 嵌入速度 | 批量处理 | 5-10倍提升 |
| LLM速度 | 使用小模型+Rerank | 成本降低50% |
| 内存占用 | 量化嵌入向量 | 减少75% |
| 分块精度 | 使用tiktoken精确计数 | 减少上下文窗口溢出 |

---

## 📁 项目结构

```
rag-knowledge-base/
├── rag_knowledge_base.py    # 核心RAG引擎（~550行）
├── app.py                   # Streamlit Web界面（~280行）
├── config.yml               # 配置文件（模型、分块、检索参数）
├── requirements.txt         # Python依赖
├── .env                     # API密钥（不提交git）
├── .env.example             # 环境变量模板
├── .streamlit/
│   └── config.toml          # Streamlit配置（上传大小限制）
├── docs/
│   └── README.md            # 本文档
├── data/                    # 数据存储目录
├── vector_db/               # 向量数据库目录
└── docker/
    ├── Dockerfile           # Docker镜像
    ├── docker-compose.yml   # 一键部署
    └── README.md            # 部署指南
```

---

## 🔗 相关资源

- [LangChain文档](https://python.langchain.com/)
- [ChromaDB文档](https://docs.trychroma.com/)
- [RAG技术详解](https://www.pinecone.io/learn/series/langchain/langchain-rag/)
- [sentence-transformers](https://www.sbert.net/)

---

## 📞 联系

如有问题或建议，欢迎提交Issue或联系开发者。

---

*最后更新: 2026-05-30*
