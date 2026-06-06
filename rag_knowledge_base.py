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
    def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """文本分块 - 使用 list 拼接避免 O(n²) 性能问题"""
        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_parts = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len < chunk_size:
                current_parts.append(para)
                current_len += para_len + 2  # +2 for "\n\n"
            else:
                if current_parts:
                    chunks.append("\n\n".join(current_parts).strip())
                # 处理长段落
                if para_len > chunk_size:
                    # 按句子分割（支持中英文标点）
                    sentences = re.split(r'(?<=[。！？.!?])\s*', para)
                    current_parts = []
                    current_len = 0
                    for sent in sentences:
                        if not sent.strip():
                            continue
                        if current_len + len(sent) < chunk_size:
                            current_parts.append(sent)
                            current_len += len(sent)
                        else:
                            if current_parts:
                                chunks.append("".join(current_parts).strip())
                            current_parts = [sent]
                            current_len = len(sent)
                else:
                    current_parts = [para]
                    current_len = para_len

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())

        # 添加重叠
        if chunk_overlap > 0 and len(chunks) > 1:
            final_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                overlap = chunks[i-1][-chunk_overlap:]
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
        """模型内存优化：GPU 可用时转为半精度，减少 ~50% 内存"""
        if self.model is None:
            return
        import torch
        if torch.cuda.is_available():
            self.model = self.model.half()
            logger.debug("模型已转为半精度 (FP16)，内存占用减少 ~50%")
        else:
            logger.debug("CPU 模式运行，跳过半精度转换")

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
        logger.debug(f"生成 {len(texts)} 个文档嵌入向量...")
        return self.model.encode(texts).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """生成查询嵌入向量"""
        if self.model is None:
            self.load_model()
        # bge 模型需要加检索指令前缀以获得更好效果
        query = "为这个句子生成表示以用于检索相关文章：" + query
        return self.model.encode([query])[0].tolist()


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
        """初始化所有组件"""
        logger.info("🔄 初始化RAG知识库...")
        self.embedding_manager.load_model()
        self.vector_store.initialize()
        self.is_initialized = True
        logger.info("✅ 初始化完成")
    
    def add_document(self, file_path: str, collection_name: str = "default"):
        """添加文档到知识库"""
        if not self.is_initialized:
            self.initialize()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"📄 处理文档: {path.name}")

        # 加载文档
        text = self.processor.load_document(path)

        # 文本分块
        chunks = self.processor.split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        logger.info(f"📄 {path.name}: 分成 {len(chunks)} 个块")

        # 生成嵌入
        embeddings = self.embedding_manager.embed_documents(chunks)

        # 添加元数据
        metadatas = [{
            "source": str(path),
            "filename": path.name,
            "collection": collection_name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        # 添加到向量库
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
                    similarity = 1 - dist
                    # 过滤低相似度结果，减少无效上下文
                    if similarity >= Config.SIMILARITY_THRESHOLD:
                        contexts.append({
                            "content": doc,
                            "metadata": meta,
                            "similarity": similarity
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
