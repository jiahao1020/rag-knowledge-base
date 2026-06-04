# RAG知识库 - Docker部署方案

## 快速部署

### 1. 构建镜像

```bash
docker build -t rag-knowledge-base .
```

### 2. 运行容器

```bash
docker run -d \
  --name rag-kb \
  -p 8501:8501 \
  -v ./data:/app/data \
  -v ./docs:/app/docs \
  -e OPENAI_API_KEY=your-key \
  rag-knowledge-base
```

### 3. 访问Web界面

浏览器打开: http://localhost:8501

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用
COPY . .

# 暴露端口
EXPOSE 8501

# 启动
CMD ["python", "app.py"]
```

---

## docker-compose.yml

```yaml
version: '3.8'

services:
  rag-kb:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./docs:/app/docs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    restart: unless-stopped

  # 可选：Qdrant向量数据库
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant-storage:/qdrant/storage
    restart: unless-stopped
```

---

## 部署到云服务器

### 1. 准备服务器

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install docker.io docker-compose -y
```

### 2. 上传代码

```bash
scp -r rag-knowledge-base user@server:/opt/
```

### 3. 部署

```bash
cd /opt/rag-knowledge-base
docker-compose up -d
```

### 4. 配置Nginx（可选）

```nginx
server {
    listen 80;
    server_name rag.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 安全建议

1. **API密钥管理**：使用环境变量，不要硬编码
2. **访问控制**：添加用户名密码认证
3. **HTTPS**：使用Let's Encrypt免费证书
4. **数据备份**：定期备份data目录
5. **日志监控**：记录访问日志
