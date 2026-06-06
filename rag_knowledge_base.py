#!/usr/bin/env python3
"""
RAG知识库系统 - 轻量级实现
支持：PDF/Markdown/TXT文档上传、向量检索、LLM生成回答

依赖安装:
    pip install langchain langchain-community chromadb sentence-transformers tiktoken pypdf python-docx markdown beautifulsoup4

快速开始:
    python rag_knowledge_base.py --help
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# 加载 .env 文件（如存在），使 LLM API key 等环境变量生效
from dotenv import load_dotenv
load_dotenv()

# ==================== 日志配置 ====================

import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "rag_kb") -> logging.Logger:
    """配置日志格式，支持 LOG_LEVEL 环境变量控制级别"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 终端输出（stderr）
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件输出（自动轮转，单个文件最大 5MB，保留 3 个备份）
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "rag_kb.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# ==================== 配置 ====================

class Config:
    """系统配置 - 优先从 config.yml 加载，不存在则使用默认值"""
    # 项目根目录（固定，不放入配置文件）
    PROJECT_ROOT = Path(__file__).parent

    # 数据存储目录（固定）
    DATA_DIR = PROJECT_ROOT / "data"
    DOCS_DIR = PROJECT_ROOT / "docs"
    DB_DIR = PROJECT_ROOT / "vector_db"

    # 确保目录存在
    for d in [DATA_DIR, DOCS_DIR, DB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 配置文件路径
    CONFIG_FILE = PROJECT_ROOT / "config.yml"

    # 默认配置
    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o"
    LLM_BASE_URL = None  # 默认为官方 API，国内模型改为此地址
    VECTOR_DB = "chroma"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7

    _loaded = False

    @classmethod
    def load(cls, force: bool = False):
        """从 YAML 文件加载配置"""
        if cls._loaded and not force:
            return
        cls._loaded = True

        yml_path = cls.CONFIG_FILE
        if not yml_path.exists():
            return

        try:
            import yaml
            with open(yml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                return

            mapping = {
                "embedding_model": "EMBEDDING_MODEL",
                "llm_provider": "LLM_PROVIDER",
                "llm_model": "LLM_MODEL",
                "llm_base_url": "LLM_BASE_URL",
                "vector_db": "VECTOR_DB",
                "chunk_size": "CHUNK_SIZE",
                "chunk_overlap": "CHUNK_OVERLAP",
                "top_k": "TOP_K",
                "similarity_threshold": "SIMILARITY_THRESHOLD",
            }
            for yaml_key, attr in mapping.items():
                if yaml_key in data and data[yaml_key] is not None:
                    setattr(cls, attr, data[yaml_key])

            logger.info(f"✅ 已加载配置: {yml_path}")
        except ImportError:
            logger.warning("⚠️  pyyaml 未安装，使用默认配置 (pip install pyyaml)")
        except Exception as e:
            logger.warning(f"⚠️  配置文件加载失败 ({e})，使用默认配置")


# 启动时自动加载配置
Config.load()


# ==================== 文档处理 ====================

class DocumentProcessor:
    """文档处理类 - 支持多种格式"""
    
    @staticmethod
    def load_document(file_path: Path) -> str:
        """加载文档内容"""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return DocumentProcessor._load_pdf(file_path)
        elif suffix in [".md", ".markdown"]:
            return DocumentProcessor._load_markdown(file_path)
        elif suffix == ".txt":
            return DocumentProcessor._load_text(file_path)
        elif suffix == ".docx":
            return DocumentProcessor._load_docx(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return DocumentProcessor._load_excel(file_path)
        else:
            raise ValueError(f"不支持的文档格式: {suffix}")

    @staticmethod
    def load_url(url: str) -> str:
        """加载在线文档"""
        try:
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(url)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        except ImportError:
            raise ImportError("请安装langchain-community和beautifulsoup4: pip install langchain-community beautifulsoup4")
    
    @staticmethod
    def _load_pdf(file_path: Path) -> str:
        """加载PDF文档"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text())
            return "\n\n".join(texts)
        except ImportError:
            raise ImportError("请安装pypdf: pip install pypdf")
    
    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """加载Markdown文档"""
        return file_path.read_text(encoding="utf-8")
    
    @staticmethod
    def _load_text(file_path: Path) -> str:
        """加载TXT文档"""
        return file_path.read_text(encoding="utf-8")
    
    @staticmethod
    def _load_docx(file_path: Path) -> str:
        """加载Word文档"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            texts = [para.text for para in doc.paragraphs]
            return "\n".join(texts)
        except ImportError:
            raise ImportError("请安装python-docx: pip install python-docx")

    @staticmethod
    def _load_excel(file_path: Path) -> str:
        """加载Excel文档（备选：整表文本）"""
        rows_data = DocumentProcessor._load_excel_as_rows(file_path)
        parts = []
        for sheet_name, row_idx, row_text, _ in rows_data:
            parts.append(f"【工作表: {sheet_name} | 行: {row_idx}】\n{row_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _detect_excel_type(file_path: Path) -> bool:
        """检测Excel是否需要清洗入库
        返回 True = 需要清洗（结构化数据，行间有依赖）
               False = 逐行存储即可（普通表格）
        """
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            needs_clean = False
            for sheet_name in wb.sheetnames[:3]:  # 只看前3个sheet
                ws = wb[sheet_name]
                # 读取表头
                for cell in next(ws.iter_rows(values_only=True), []):
                    header = str(cell) if cell else ""
                    # 关键词检测
                    clean_keywords = ["脚本", "SQL", "查询", "修复", "任务类别",
                                      "sql", "脚本", "语句"]
                    if any(k in header for k in clean_keywords):
                        needs_clean = True
                        break
                if needs_clean:
                    break

                # 检测前10行是否有SQL关键词
                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=2):
                    if row_idx > 10:
                        break
                    for cell in row:
                        val = str(cell) if cell else ""
                        import re
                        if re.search(r'\b(update|insert into|delete from|alter table|create table|select\s.*from)\b',
                                     val, re.IGNORECASE):
                            needs_clean = True
                            break
                    if needs_clean:
                        break

            wb.close()
            return needs_clean
        except ImportError:
            return False

    @staticmethod
    def _load_excel_as_rows(file_path: Path) -> List[tuple]:
        """逐行读取Excel，返回 [(sheet_name, row_index, row_text, metadata_dict), ...]

        每行独立作为一个条目，适合结构化数据（如：类别 + 查询脚本 + SQL）。
        row_text 带列名前缀，便于LLM理解也利于向量检索。
        metadata_dict 包含原始列值，可用于过滤。
        """
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            rows = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                # 读取表头作为列名
                headers = []
                for cell in next(ws.iter_rows(values_only=True), []):
                    headers.append(str(cell) if cell is not None else f"列{len(headers)+1}")

                # 逐行读取，每行作为一个独立条目
                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=2):
                    cells = [str(c) if c is not None else "" for c in row]
                    if not any(c.strip() for c in cells):
                        continue

                    # 构建带列名的文本：【列名】值 | 【列名】值 | ...
                    labeled_parts = []
                    for i, val in enumerate(cells):
                        label = headers[i] if i < len(headers) else f"列{i+1}"
                        if val.strip():
                            labeled_parts.append(f"【{label}】{val}")
                    row_text = "\n".join(labeled_parts)

                    # 元数据：保留原始列值（取前3列作为关键字段方便过滤，最多10列避免元数据过大）
                    meta = {"sheet": sheet_name, "row": row_idx}
                    for i, val in enumerate(cells[:10]):
                        label = headers[i] if i < len(headers) else f"col{i}"
                        if val.strip():
                            meta[label] = val.strip()

                    rows.append((sheet_name, row_idx, row_text, meta))
            wb.close()
            return rows
        except ImportError:
            raise ImportError("请安装 openpyxl: pip install openpyxl")

    @staticmethod
    def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 80) -> List[str]:
        """递归分块：按 Markdown 标题分段，tiktoken 精确计数"""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            def count_tokens(s):
                return len(enc.encode(s, disallowed_special=()))
        except ImportError:
            def count_tokens(s):
                return int(len(s) / 1.5)

        paragraphs = text.split("\n\n")
        chunks = []
        current_parts = []
        current_len = 0

        def flush():
            nonlocal current_parts, current_len
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_len = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)
            is_heading = re.match(r"^#{1,6}\s", para.strip())

            if is_heading and current_parts:
                flush()

            if not current_parts:
                if para_tokens > chunk_size:
                    # 段落本身超长，按句子拆分
                    sentences = re.split(r"(?<=[。！？.!?])\s*", para)
                    for sent in sentences:
                        if not sent.strip():
                            continue
                        sent_tokens = count_tokens(sent)
                        # 如果句子仍然超长，按字符拆分
                        if sent_tokens > chunk_size:
                            for char in sent:
                                if not char.strip():
                                    continue
                                char_tokens = count_tokens(char)
                                if current_parts:
                                    current_parts.append(char)
                                    current_len += char_tokens
                                else:
                                    current_parts = [char]
                                    current_len = char_tokens
                                # 字符拆分时也要检查是否超过 chunk_size
                                if current_len >= chunk_size:
                                    flush()
                        else:
                            if current_parts:
                                current_parts.append(sent)
                                current_len += sent_tokens
                            else:
                                current_parts = [sent]
                                current_len = sent_tokens
                else:
                    current_parts = [para]
                    current_len = para_tokens
            elif current_len + para_tokens <= chunk_size:
                current_parts.append(para)
                current_len += para_tokens
            else:
                flush()
                if para_tokens > chunk_size * 0.8:
                    sentences = re.split(r"(?<=[。！？.!?])\s*", para)
                    for sent in sentences:
                        if not sent.strip():
                            continue
                        sent_tokens = count_tokens(sent)
                        if sent_tokens > chunk_size:
                            for char in sent:
                                if not char.strip():
                                    continue
                                char_tokens = count_tokens(char)
                                if current_parts:
                                    current_parts.append(char)
                                    current_len += char_tokens
                                else:
                                    current_parts = [char]
                                    current_len = char_tokens
                                # 字符拆分时也要检查是否超过 chunk_size
                                if current_len >= chunk_size:
                                    flush()
                        else:
                            if current_parts:
                                current_parts.append(sent)
                                current_len += sent_tokens
                            else:
                                current_parts = [sent]
                                current_len = sent_tokens
                else:
                    current_parts = [para]
                    current_len = para_tokens

        flush()

        if chunk_overlap > 0 and len(chunks) > 1:
            final_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                overlap_chars = min(len(prev), chunk_overlap * 3)
                overlap = prev[-overlap_chars:]
                final_chunks.append(overlap + chunks[i])
            return final_chunks

        return chunks


# ==================== 向量化 ====================

class EmbeddingManager:
    """嵌入向量管理"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """加载嵌入模型，优先从 ModelScope 下载（国内网络友好）"""
        try:
            from sentence_transformers import SentenceTransformer

            # 先检查本地缓存
            model_path = self._find_local_cache()
            if model_path:
                self.model = SentenceTransformer(model_path)
                logger.info(f"✅ 从本地缓存加载嵌入模型: {self.model_name}")
                self._optimize_model()
                return

            # 优先从 ModelScope 下载
            try:
                self._load_from_modelscope()
                self._optimize_model()
                return
            except Exception:
                logger.warning("⚠️  ModelScope 下载失败，尝试从 HuggingFace 下载...")

            # 最后尝试 HuggingFace
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ 从 HuggingFace 加载嵌入模型: {self.model_name}")
            self._optimize_model()

        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

    def _optimize_model(self):
        """模型优化：GPU 半精度 + CPU 线程优化"""
        if self.model is None:
            return
        import torch
        if torch.cuda.is_available():
            self.model = self.model.half()
            logger.info("✅ 模型已转为半精度 (FP16)，内存占用减少 ~50%")
        else:
            # CPU 模式：优化线程数以充分利用多核
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            logger.info(f"✅ CPU 线程数已优化: {torch.get_num_threads()}")

    def _find_local_cache(self) -> Optional[str]:
        """查找本地是否已有缓存的模型"""
        import os
        # HuggingFace cache 目录结构
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = self.model_name.replace("/", "--")
        # 检查 snapshots 下是否有文件
        snapshots_dir = cache_dir / f"models--{model_id}" / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir() and any(snapshot.iterdir()):
                    return str(snapshot)
        # 检查 sentence-transformers 缓存
        st_cache = Path.home() / ".cache" / "torch" / "sentence_transformers" / self.model_name
        if st_cache.exists():
            return str(st_cache)
        return None

    def _load_from_modelscope(self):
        """从 ModelScope 下载并加载模型"""
        try:
            from modelscope import snapshot_download
            from sentence_transformers import SentenceTransformer

            model_dir = snapshot_download(self.model_name)
            self.model = SentenceTransformer(model_dir)
            logger.info(f"✅ 从 ModelScope 加载嵌入模型: {self.model_name}")
        except ImportError:
            raise ImportError(
                "请安装 modelscope: pip install modelscope\n"
                "或手动下载模型: python -c \"from modelscope import snapshot_download; "
                "snapshot_download('BAAI/bge-large-zh-v1.5')\""
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文档嵌入向量"""
        if self.model is None:
            self.load_model()
        logger.info(f"⏳ 正在为 {len(texts)} 个文档块生成嵌入向量（首次较慢，请耐心等待）...")
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        """生成查询嵌入向量"""
        if self.model is None:
            self.load_model()
        # bge 模型需要加检索指令前缀以获得更好效果
        query = "为这个句子生成表示以用于检索相关文章：" + query
        return self.model.encode([query], show_progress_bar=False)[0].tolist()


# ==================== 向量数据库 ====================

class VectorStore:
    """向量数据库管理"""
    
    def __init__(self, db_path: Path = Config.DB_DIR):
        self.db_path = db_path
        self.collection = None
    
    def initialize(self):
        """初始化向量数据库"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 持久化存储
            client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = client.get_or_create_collection(
                name="knowledge_base",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 12,           # 默认16，降低连接数减少内存
                    "hnsw:construction_ef": 100,  # 构建精度
                    "hnsw:search_ef": 20,   # 默认10，适当提高检索召回
                }
            )
            logger.info(f"✅ 向量数据库初始化: {self.db_path}")
        except ImportError:
            raise ImportError("请安装chromadb: pip install chromadb")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict] = None):
        """添加文档到向量库（写入失败自动回滚）"""
        if self.collection is None:
            self.initialize()

        # 生成ID
        ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
               for i in range(len(texts))]

        # 默认元数据
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ 添加了 {len(texts)} 个文档块")
        except Exception as e:
            # 写入失败时清理部分写入的数据
            try:
                self.collection.delete(ids=ids)
                logger.error(f"写入失败，已回滚 {len(texts)} 个文档块: {e}")
            except Exception:
                logger.error(f"写入失败且回滚异常: {e}")
            raise e
    
    def search(self, query_embedding: List[float], top_k: int = 5):
        """向量检索"""
        if self.collection is None:
            self.initialize()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        if self.collection is None:
            return {"count": 0}
        return {
            "count": self.collection.count(),
            "space": self.collection.metadata.get("hnsw:space", "cosine")
        }


# ==================== LLM生成 ====================

class LLMGenerator:
    """LLM回答生成"""

    def __init__(self, provider: str = Config.LLM_PROVIDER,
                 model: str = Config.LLM_MODEL):
        self.provider = provider
        self.model = model
        self._llm_cache = {}  # 缓存 LLM 实例避免重复创建

    def _get_llm(self):
        """获取（或创建）缓存的 LLM 实例"""
        cache_key = (self.provider, self.model, Config.LLM_BASE_URL)
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        try:
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI
                self._llm_cache[cache_key] = ChatOpenAI(
                    model=self.model, temperature=0.7,
                    base_url=Config.LLM_BASE_URL,
                )
            elif self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self._llm_cache[cache_key] = ChatAnthropic(model=self.model, temperature=0.7)
            elif self.provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm_cache[cache_key] = ChatGoogleGenerativeAI(model=self.model, temperature=0.7)
            elif self.provider == "local":
                from langchain_community.chat_models import ChatOllama
                self._llm_cache[cache_key] = ChatOllama(model="llama3", temperature=0.7)
            else:
                raise ValueError(f"不支持的LLM提供商: {self.provider}")
        except ImportError as e:
            pkg_map = {
                "openai": "langchain-openai",
                "anthropic": "langchain-anthropic",
                "google": "langchain-google-genai",
                "local": "langchain-community",
            }
            raise ImportError(f"请安装 {pkg_map.get(self.provider, 'langchain')}: pip install {pkg_map.get(self.provider, 'langchain')}") from e

        return self._llm_cache[cache_key]

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """生成回答"""
        if not context:
            return "未检索到相关文档，请先上传文档到知识库。"

        # 构建上下文（修复 metadata 取值路径）
        context_text = "\n\n".join([
            f"来源: {c['metadata'].get('filename', c['metadata'].get('source', 'unknown'))}\n内容: {c['content']}"
            for c in context
        ])

        prompt = f"""你是一位知识助手，请根据以下检索到的信息回答问题。
如果信息不足，请诚实说明。

【检索到的信息】
{context_text}

【用户问题】
{query}

【回答要求】
1. 基于检索到的信息回答
2. 引用信息来源
3. 如果信息不足，说明需要补充什么信息
4. 回答简洁明了

【回答】
"""

        try:
            from langchain_core.messages import HumanMessage
            llm = self._get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API（保留，兼容外部直接调用）"""
        from langchain_core.messages import HumanMessage
        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """调用Anthropic API"""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            
            llm = ChatAnthropic(model=self.model, temperature=0.7)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except ImportError:
            raise ImportError("请安装langchain-anthropic: pip install langchain-anthropic")
    
    def _call_google(self, prompt: str) -> str:
        """调用Google Gemini API"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatGoogleGenerativeAI(model=self.model, temperature=0.7)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except ImportError:
            raise ImportError("请安装langchain-google-genai: pip install langchain-google-genai")
    
    def _call_local(self, prompt: str) -> str:
        """调用本地模型（Ollama等）"""
        try:
            from langchain_community.chat_models import ChatOllama
            from langchain_core.messages import HumanMessage
            
            llm = ChatOllama(model="llama3", temperature=0.7)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except ImportError:
            raise ImportError("请安装langchain-community和Ollama")


# ==================== RAG知识库主类 ====================

class RAGKnowledgeBase:
    """RAG知识库主类"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.llm_generator = LLMGenerator()
        self.is_initialized = False
    
    def initialize(self):
        """初始化所有组件（延迟加载嵌入模型，加快启动速度）"""
        logger.info("🔄 初始化RAG知识库...")
        self.vector_store.initialize()  # 轻量，仅连接 ChromaDB
        self.is_initialized = True
        logger.info("✅ 初始化完成（嵌入模型将在首次使用时加载）")
    
    def add_document(self, file_path: str, collection_name: str = "default"):
        """添加文档到知识库"""
        if not self.is_initialized:
            self.initialize()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"📄 处理文档: {path.name}")

        # Excel 文件：行级处理（结构化数据，每行一个独立条目）
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return self._add_excel_rows(path, collection_name)

        # 其他格式：传统文本流水线
        text = self.processor.load_document(path)
        chunks = self.processor.split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        logger.info(f"📄 {path.name}: 分成 {len(chunks)} 个块")

        embeddings = self.embedding_manager.embed_documents(chunks)
        metadatas = [{
            "source": str(path),
            "filename": path.name,
            "collection": collection_name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        self.vector_store.add_documents(chunks, embeddings, metadatas)
        return len(chunks)

    def _add_excel_rows(self, path: Path, collection_name: str) -> int:
        """Excel 智能入库：自动检测类型，决定清洗或逐行存储"""
        if self.processor._detect_excel_type(path):
            logger.info(f"🔍 {path.name}：检测为结构化数据，启动智能清洗")
            return self._add_excel_rows_cleaned(path, collection_name)
        else:
            logger.info(f"📋 {path.name}：检测为普通表格，逐行存储")
            return self._add_excel_rows_raw(path, collection_name)

    def _add_excel_rows_raw(self, path: Path, collection_name: str) -> int:
        """普通表格：逐行直接存储"""
        rows = self.processor._load_excel_as_rows(path)
        if not rows:
            logger.warning(f"⚠️  Excel 文件为空: {path.name}")
            return 0

        chunks = []
        metadatas = []
        for sheet_name, row_idx, row_text, row_meta in rows:
            chunks.append(row_text)
            metadatas.append({
                "source": str(path),
                "filename": path.name,
                "collection": collection_name,
                "sheet": sheet_name,
                "row": row_idx,
            })

        logger.info(f"📊 {path.name}: {len(chunks)} 行")
        embeddings = self.embedding_manager.embed_documents(chunks)
        self.vector_store.add_documents(chunks, embeddings, metadatas)
        return len(chunks)

    def _add_excel_rows_cleaned(self, path: Path, collection_name: str) -> int:
        """结构化数据：自动按任务类别分组，合并为自然语言描述"""
        import re

        rows = self.processor._load_excel_as_rows(path)
        if not rows:
            logger.warning(f"⚠️  Excel 文件为空: {path.name}")
            return 0

        logger.info(f"📊 {path.name}: 共 {len(rows)} 行数据")

        # 按 sheet 分组（天然支持多 Sheet）
        from collections import defaultdict
        sheets_data = defaultdict(list)
        for sheet_name, row_idx, row_text, row_meta in rows:
            sheets_data[sheet_name].append((row_idx, row_text, row_meta))

        chunks = []
        metadatas = []

        for sheet_name, sheet_rows in sheets_data.items():
            # 按行号排序
            sheet_rows.sort(key=lambda x: x[0])

            # 遍历行，按有"任务类别"的行做分组
            groups = []
            current_group = []

            for row_idx, row_text, row_meta in sheet_rows:
                has_task = bool(row_meta.get("任务类别", ""))
                if has_task and current_group:
                    groups.append(current_group)
                    current_group = []
                current_group.append((row_idx, row_text, row_meta))

            if current_group:
                groups.append(current_group)

            # 处理每个分组：合并为自然语言
            for group in groups:
                task_category = ""
                query_script = ""
                fix_items = []
                fix_scripts = []
                table_mappings = {}

                for row_idx, row_text, row_meta in group:
                    tc = row_meta.get("任务类别", "")
                    qs = row_meta.get("前置查询脚本", "")
                    fi = row_meta.get("修复项", "")
                    fs = row_meta.get("修复脚本", "")

                    if tc and not task_category:
                        task_category = tc
                    if qs and not query_script:
                        query_script = qs
                    if fi:
                        fix_items.append(fi)
                    if fs:
                        fix_scripts.append(fs)

                    # 收集表名列（非已知列且值不空的列）
                    known_cols = {"任务类别", "前置查询脚本", "修复项", "修复脚本", "修复说明"}
                    for k, v in row_meta.items():
                        col_name = k
                        if col_name not in known_cols and v.strip():
                            table_mappings[col_name] = v.strip()

                # 跳过纯表头行
                if not task_category and not query_script and not fix_items:
                    continue

                # 构建自然语言描述
                parts = []
                if task_category:
                    parts.append(f"任务类别：{task_category}。")
                if query_script:
                    parts.append(f"前置查询脚本：{query_script}。")

                # 修复项 + 修复脚本
                for fi in fix_items:
                    # 检查同一分组里有没有对应的修复脚本（表名列含 update/delete/insert）
                    found_script = False
                    for table_name, table_value in table_mappings.items():
                        if re.search(r'(update|insert|delete|alter|create|drop)\s', table_value, re.IGNORECASE):
                            parts.append(f"修复项：{fi}。修复脚本（表：{table_name}）：{table_value}。")
                            found_script = True
                        else:
                            # 非 SQL 的值当成字段名
                            parts.append(f"修复字段（表：{table_name}）：{table_value}。")

                # 涉及的表名
                table_names = [t for t in table_mappings.keys()
                               if not re.search(r'(update|insert|delete)', table_mappings[t], re.IGNORECASE)]
                if table_names:
                    parts.append(f"涉及表：{'、'.join(table_names)}。")

                chunk_text = "".join(parts)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    meta = {
                        "source": str(path),
                        "filename": path.name,
                        "collection": collection_name,
                        "sheet": sheet_name,
                        "task_category": task_category or "未分类",
                    }
                    metadatas.append(meta)

        if not chunks:
            logger.warning(f"⚠️  {path.name}：清洗后无有效数据")
            return 0

        logger.info(f"🧹 {path.name}：清洗后生成 {len(chunks)} 条自然语言描述")
        embeddings = self.embedding_manager.embed_documents(chunks)
        self.vector_store.add_documents(chunks, embeddings, metadatas)
        return len(chunks)

    def add_url(self, url: str, collection_name: str = "default"):
        """添加在线文档到知识库"""
        if not self.is_initialized:
            self.initialize()

        logger.info(f"🌐 加载在线文档: {url}")

        # 加载在线文档
        text = self.processor.load_url(url)
        if not text.strip():
            raise ValueError(f"无法从 {url} 获取内容")

        # 文本分块
        chunks = self.processor.split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        logger.info(f"   分成 {len(chunks)} 个块")

        # 生成嵌入
        embeddings = self.embedding_manager.embed_documents(chunks)

        # 添加元数据
        metadatas = [{
            "source": url,
            "filename": url,
            "collection": collection_name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        # 添加到向量库
        self.vector_store.add_documents(chunks, embeddings, metadatas)

        return len(chunks)
    
    def add_directory(self, dir_path: str, pattern: str = "*.md", 
                      collection_name: str = "default"):
        """添加目录下的所有文档"""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        files = list(path.glob(pattern))
        logger.info(f"📁 找到 {len(files)} 个文件")
        
        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.add_document(str(file_path), collection_name)
                total_chunks += chunks
            except Exception as e:
                logger.error(f"⚠️  处理 {file_path.name} 失败: {e}")
        
        return total_chunks
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """查询知识库"""
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"🔍 查询: {question}")
        
        # 生成查询嵌入
        query_embedding = self.embedding_manager.embed_query(question)
        
        # 向量检索
        results = self.vector_store.search(query_embedding, top_k)
        
        # 构建上下文
        contexts = []
        if results and results.get('documents'):
            for docs, metadatas, distances in zip(
                results['documents'], 
                results['metadatas'], 
                results['distances']
            ):
                for doc, meta, dist in zip(docs, metadatas, distances):
                    contexts.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity": 1 - dist
                    })
        
        logger.info(f"   检索到 {len(contexts)} 个相关文档块")
        
        # 生成回答
        answer = self.llm_generator.generate_answer(question, contexts)
        
        return {
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        db_stats = self.vector_store.get_stats()
        return {
            "document_count": db_stats.get("count", 0),
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_provider": Config.LLM_PROVIDER,
            "llm_model": Config.LLM_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP
        }
    
    def export(self, output_path: str = None):
        """导出知识库"""
        if output_path is None:
            output_path = Config.DATA_DIR / f"kb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats = self.get_stats()
        export_data = {
            "stats": stats,
            "exported_at": datetime.now().isoformat(),
            "config": {
                "embedding_model": Config.EMBEDDING_MODEL,
                "llm_provider": Config.LLM_PROVIDER,
                "llm_model": Config.LLM_MODEL,
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 知识库已导出: {output_path}")
        return output_path


# ==================== CLI 接口 ====================

def main():
    parser = argparse.ArgumentParser(description="RAG知识库系统")
    parser.add_argument("--command", "-c", choices=["add", "query", "stats", "export"],
                        help="执行命令")
    parser.add_argument("--file", "-f", help="要添加的文件路径")
    parser.add_argument("--url", "-u", help="要添加的在线文档URL")
    parser.add_argument("--dir", "-d", help="要添加的目录路径")
    parser.add_argument("--question", "-q", help="查询问题")
    parser.add_argument("--pattern", "-p", default="*.md", help="文件匹配模式")
    parser.add_argument("--output", "-o", help="导出路径")
    
    args = parser.parse_args()
    
    kb = RAGKnowledgeBase()
    
    if args.command == "add":
        if args.file:
            kb.add_document(args.file)
        elif args.url:
            kb.add_url(args.url)
        elif args.dir:
            kb.add_directory(args.dir, args.pattern)
        else:
            logger.error("❌ 请指定 --file、--url 或 --dir")

    elif args.command == "query":
        if not args.question:
            logger.error("❌ 请指定 --question")
            return
        result = kb.query(args.question)
        print("\n" + "="*60)
        print("回答:")
        print(result["answer"])
        print("="*60)
        print("\n引用来源:")
        for i, ctx in enumerate(result["contexts"]):
            print(f"\n{i+1}. [{ctx['similarity']:.2f}] {ctx['metadata'].get('filename', 'unknown')}")
    
    elif args.command == "stats":
        stats = kb.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.command == "export":
        kb.export(args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
