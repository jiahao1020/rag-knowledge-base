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
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# 加载 .env 文件（如存在），使 LLM API key 等环境变量生效
from dotenv import load_dotenv
load_dotenv()

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
    def load(cls):
        """从 YAML 文件加载配置"""
        if cls._loaded:
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

            print(f"✅ 已加载配置: {yml_path}")
        except ImportError:
            print("⚠️  pyyaml 未安装，使用默认配置 (pip install pyyaml)")
        except Exception as e:
            print(f"⚠️  配置文件加载失败 ({e})，使用默认配置")


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
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
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
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except ImportError:
            raise ImportError("请安装python-docx: pip install python-docx")
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """文本分块"""
        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 处理长段落
                if len(para) > chunk_size:
                    # 按句子分割
                    sentences = para.split(". ")
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) < chunk_size:
                            current_chunk += sent + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent + ". "
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
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
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ 加载嵌入模型: {self.model_name}")
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文档嵌入向量"""
        if self.model is None:
            self.load_model()
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
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ 向量数据库初始化: {self.db_path}")
        except ImportError:
            raise ImportError("请安装chromadb: pip install chromadb")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                      metadatas: List[Dict] = None):
        """添加文档到向量库"""
        if self.collection is None:
            self.initialize()
        
        # 生成ID
        ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
               for i in range(len(texts))]
        
        # 默认元数据
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ 添加了 {len(texts)} 个文档块")
    
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
    
    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """生成回答"""
        # 构建上下文
        context_text = "\n\n".join([
            f"来源: {m.get('source', 'unknown')}\n内容: {d}"
            for d, m in zip(context, [c.get('metadatas', [{}])[0] for c in context])
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
        
        # 调用LLM（这里以OpenAI为例，可切换其他提供商）
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "google":
            return self._call_google(prompt)
        elif self.provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatOpenAI(
                model=self.model,
                temperature=0.7,
                base_url=Config.LLM_BASE_URL,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except ImportError:
            raise ImportError("请安装langchain-openai: pip install langchain-openai")
    
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
        print("🔄 初始化RAG知识库...")
        self.embedding_manager.load_model()
        self.vector_store.initialize()
        self.is_initialized = True
        print("✅ 初始化完成")
    
    def add_document(self, file_path: str, collection_name: str = "default"):
        """添加文档到知识库"""
        if not self.is_initialized:
            self.initialize()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        print(f"📄 处理文档: {path.name}")

        # 加载文档
        text = self.processor.load_document(path)

        # 文本分块
        chunks = self.processor.split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        print(f"   分成 {len(chunks)} 个块")

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

        print(f"🌐 加载在线文档: {url}")

        # 加载在线文档
        text = self.processor.load_url(url)
        if not text.strip():
            raise ValueError(f"无法从 {url} 获取内容")

        # 文本分块
        chunks = self.processor.split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        print(f"   分成 {len(chunks)} 个块")

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
        print(f"📁 找到 {len(files)} 个文件")
        
        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.add_document(str(file_path), collection_name)
                total_chunks += chunks
            except Exception as e:
                print(f"   ⚠️ 处理 {file_path.name} 失败: {e}")
        
        return total_chunks
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """查询知识库"""
        if not self.is_initialized:
            self.initialize()
        
        print(f"🔍 查询: {question}")
        
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
                        "similarity": 1 - dist  # 转换为相似度
                    })
        
        print(f"   检索到 {len(contexts)} 个相关文档块")
        
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
        
        print(f"✅ 知识库已导出: {output_path}")
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
            print("❌ 请指定 --file、--url 或 --dir")
    
    elif args.command == "query":
        if not args.question:
            print("❌ 请指定 --question")
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
