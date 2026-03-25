# Qwen3-Embedding Service

基于 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 模型的文本向量化 API 服务，使用 FastAPI + Sentence Transformers 构建，完整实现 OpenAI Embeddings API 标准接口。

---

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [环境变量配置](#环境变量配置)
- [模型加载策略](#模型加载策略)
- [API 接口](#api-接口)
- [Docker 部署](#docker-部署)
- [systemd 服务部署](#systemd-服务部署)

---

## 功能特性

- **OpenAI 标准接口**：`POST /v1/embeddings` 和 `GET /v1/models`，可直接替换 OpenAI SDK 的 base_url
- **本地模型优先**：配置 `LOCAL_MODEL_PATH` 后从磁盘加载，无需网络，适合离线/内网环境
- **自动设备检测**：自动选择 CUDA / MPS（Apple Silicon）/ CPU
- **批量推理**：支持单条和多条文本批量向量化
- **Docker 支持**：提供 GPU 镜像和 docker-compose 配置

---

## 快速开始

### 1. 安装依赖

推荐使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv sync
```

或使用 pip：

```bash
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 按需修改 .env
```

### 3. 启动服务

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

服务启动后访问 [http://localhost:8000/docs](http://localhost:8000/docs) 查看交互式文档。

---

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_MODEL_PATH` | _(空)_ | 本地模型目录路径，**优先于 HuggingFace 下载**，支持 `~` |
| `MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace 模型 ID，仅在未配置本地路径时使用 |
| `DEVICE` | _(自动)_ | 推理设备：`cuda` / `mps` / `cpu`，留空自动检测 |
| `NORMALIZE_EMBEDDINGS` | `true` | 是否对向量进行 L2 归一化 |
| `BATCH_SIZE` | `32` | 批量推理大小 |
| `HF_HOME` | _(系统默认)_ | HuggingFace 模型缓存目录 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |

---

## 模型加载策略

服务启动时按以下优先级决定从哪里加载模型：

```
LOCAL_MODEL_PATH 已配置且目录存在
        ↓ 是
从本地磁盘加载（local_files_only=True，不访问网络）
        ↓ 否
从 HuggingFace Hub 下载（使用 MODEL_NAME）
```

**示例 `.env` 配置（本地模式）：**

```env
LOCAL_MODEL_PATH=~/models/Qwen3-Embedding-0.6B
MODEL_NAME=Qwen/Qwen3-Embedding-0.6B   # 路径无效时的回退
```

**手动下载模型到本地：**

```bash
# 方式一：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ~/models/Qwen3-Embedding-0.6B

# 方式二：使用 Python
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder='~/models')
"
```

---

## API 接口

### OpenAI 兼容接口

#### `POST /v1/embeddings`

完全兼容 OpenAI Embeddings API，支持将现有代码的 `base_url` 替换为本服务地址。

**请求：**

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "你好，世界"
  }'
```

支持数组输入：

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": ["第一段文本", "第二段文本"]
  }'
```

**响应：**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.023, -0.011, ...],
      "index": 0
    }
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

**Python SDK 用法（直接替换 OpenAI）：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 本服务不验证 key
)

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["你好", "世界"],
)
print(response.data[0].embedding)
```

#### `GET /v1/models`

列出当前加载的模型。

```bash
curl http://localhost:8000/v1/models
```

---

### 原生接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/model-info` | 模型详细信息（维度、最大长度等） |
| `POST` | `/embed` | 单条文本向量化 |
| `POST` | `/embeddings` | 批量文本向量化 |
| `POST` | `/similarity` | 计算两段文本的余弦相似度 |
| `GET` | `/docs` | Swagger 交互式文档 |

---

## Docker 部署

### 使用 docker-compose（推荐）

```bash
# 构建并启动（GPU）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

如需使用本地模型，修改 `.env`：

```env
LOCAL_MODEL_PATH=/root/models/Qwen3-Embedding-0.6B
```

`docker-compose.yml` 已将宿主机 `~/models` 挂载到容器 `/root/models`，本地模型目录会自动生效。

### 仅 CPU（无 GPU 环境）

修改 `docker-compose.yml`，移除 `deploy.resources` 块后执行：

```bash
docker compose up -d
```

---

## systemd 服务部署

适用于在 Linux 服务器上将服务注册为系统守护进程，开机自启、崩溃自动重启。

> **核心要点**：systemd 服务运行在无交互的 shell 环境中，`conda activate` / `source .venv/bin/activate` 等命令无效。正确做法是**直接使用虚拟环境内的可执行文件绝对路径**，或通过 `Environment=PATH=` 将虚拟环境注入到进程的 PATH 中。

---

### 方案一：uv 虚拟环境

#### 1. 创建虚拟环境并安装依赖

```bash
cd /opt/qwen-embedding

# 用 uv 创建 .venv 并同步依赖
uv sync

# 确认 uvicorn 路径
/opt/qwen-embedding/.venv/bin/uvicorn --version
```

#### 2. 创建 service 文件

```bash
sudo nano /etc/systemd/system/qwen-embedding.service
```

```ini
[Unit]
Description=Qwen3-Embedding API Service (uv venv)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/qwen-embedding

# 将 .venv/bin 注入 PATH，使虚拟环境生效
Environment=PATH=/opt/qwen-embedding/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

# 读取项目 .env 中的配置
EnvironmentFile=/opt/qwen-embedding/.env

# 直接调用虚拟环境内的 uvicorn，无需激活
ExecStart=/opt/qwen-embedding/.venv/bin/uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qwen-embedding
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

---

### 方案二：conda 虚拟环境

#### 1. 创建 conda 环境并安装依赖

```bash
# 创建环境
conda create -n qwen-embedding python=3.10 -y
conda activate qwen-embedding

# 安装依赖
pip install -e /opt/qwen-embedding

# 确认 conda 环境中的 uvicorn 路径
which uvicorn
# 示例输出：/opt/conda/envs/qwen-embedding/bin/uvicorn
```

#### 2. 创建 service 文件

```bash
sudo nano /etc/systemd/system/qwen-embedding.service
```

```ini
[Unit]
Description=Qwen3-Embedding API Service (conda)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/qwen-embedding

# 将 conda 环境的 bin 目录置于 PATH 最前，覆盖系统 Python
Environment=PATH=/opt/conda/envs/qwen-embedding/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
# conda 环境根目录（部分库通过此变量定位资源）
Environment=CONDA_PREFIX=/opt/conda/envs/qwen-embedding

# 读取项目 .env 中的配置
EnvironmentFile=/opt/qwen-embedding/.env

# 直接调用 conda 环境内的 uvicorn
ExecStart=/opt/conda/envs/qwen-embedding/bin/uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qwen-embedding
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

> **conda 路径说明**：默认安装路径为 `/opt/conda`（Miniconda/Anaconda 常见路径）。若你的安装路径不同，用以下命令确认：
> ```bash
> conda info --envs
> # 输出示例：qwen-embedding  /home/ubuntu/miniconda3/envs/qwen-embedding
> ```
> 将 service 文件中的 `/opt/conda/envs/qwen-embedding` 替换为实际路径。

---

### 3. 启用并启动服务

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 开机自启
sudo systemctl enable qwen-embedding

# 立即启动
sudo systemctl start qwen-embedding

# 查看运行状态
sudo systemctl status qwen-embedding
```

### 4. 查看日志

```bash
# 实时日志
sudo journalctl -u qwen-embedding -f

# 查看最近 100 行
sudo journalctl -u qwen-embedding -n 100

# 按时间过滤
sudo journalctl -u qwen-embedding --since "2026-03-25 08:00:00"
```

### 5. 常用管理命令

```bash
# 重启服务（更新代码或修改 .env 后）
sudo systemctl restart qwen-embedding

# 停止服务
sudo systemctl stop qwen-embedding

# 禁用开机自启
sudo systemctl disable qwen-embedding
```

### 6. 使用本地模型（离线部署）

在 `/opt/qwen-embedding/.env` 中配置：

```env
LOCAL_MODEL_PATH=/opt/models/Qwen3-Embedding-0.6B
MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
DEVICE=cpu
NORMALIZE_EMBEDDINGS=true
BATCH_SIZE=32
HOST=0.0.0.0
PORT=8000
```

重启服务使配置生效：

```bash
sudo systemctl restart qwen-embedding
```

启动日志中将显示：

```
Local model path: /opt/models/Qwen3-Embedding-0.6B
Loading model from local path: /opt/models/Qwen3-Embedding-0.6B
Model loaded successfully
```

### 7. GPU 服务器注意事项

若服务器安装了 NVIDIA GPU，需确保：

1. 已安装 NVIDIA 驱动和 CUDA
2. 虚拟环境中安装的是 CUDA 版本的 `torch`
3. `.env` 中设置 `DEVICE=cuda`

```bash
# 验证 GPU 可用（使用对应虚拟环境的 Python）
# uv:
/opt/qwen-embedding/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"

# conda:
/opt/conda/envs/qwen-embedding/bin/python -c "import torch; print(torch.cuda.is_available())"
```

> **注意**：embedding 模型首次加载需要较长时间，`Type=simple` 下 systemd 不会等待模型加载完毕才报告 active 状态，这是正常行为。通过 `journalctl -u qwen-embedding -f` 观察 `Model loaded successfully` 日志确认服务就绪。
