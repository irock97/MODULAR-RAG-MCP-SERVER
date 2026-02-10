# Modular RAG MCP Server

Modular RAG (Retrieval-Augmented Generation) MCP Server - 可插拔的检索增强生成服务。

## 项目概述

本项目是一个模块化的 RAG 系统，支持：
- 多模态文档处理（PDF → Markdown）
- 混合检索（Dense + Sparse + RRF）
- 可插拔的后端实现（LLM/Embedding/VectorStore）
- MCP 协议集成（支持 Copilot/Claude Desktop）
- 可观测性支持（Trace + Dashboard）

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装项目
pip install -e ".[all]"
```

### 2. 配置

编辑 `config/settings.yaml`：

```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: "${OPENAI_API_KEY}"

embedding:
  provider: openai
  model: text-embedding-3-small

vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
```

### 3. 运行

```bash
# 数据摄取
python scripts/ingest.py --collection my-docs --path ./data/documents

# 启动 MCP Server
python main.py

# 启动 Dashboard
python scripts/start_dashboard.py
```

## 项目结构

```
modular-rag-mcp-server/
├── config/              # 配置文件
│   ├── settings.yaml
│   └── prompts/
├── src/                 # 源代码
│   ├── mcp_server/     # MCP Server 层
│   ├── core/           # Core 层
│   ├── ingestion/      # Ingestion Pipeline
│   ├── libs/           # 可插拔抽象层
│   └── observability/  # 可观测性
├── tests/              # 测试
├── scripts/            # 脚本
└── main.py             # 入口
```

## 文档

- [开发规范](DEV_SPEC.md)
- [API 文档](docs/)

## 许可证

MIT
