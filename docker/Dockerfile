FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

# 复制应用文件
COPY rag_knowledge_base.py .
COPY app.py .
COPY docs/ ./docs/

# 创建数据目录
RUN mkdir -p /app/data /app/vector_db

# 暴露端口
EXPOSE 8501

# 环境变量
ENV OPENAI_API_KEY=
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
