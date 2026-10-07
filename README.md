# 📚 企业知识库问答（RAG）

基于 **FastAPI + LangChain + ChromaDB** 的企业知识库问答系统。上传文档后，即可基于文档内容进行自然语言问答，答案附带引用来源。

一个完整的、可直接部署的 RAG 落地项目，适合作为学习参考或个人作品。

## ✨ 功能

- 📤 文档上传：支持 PDF / TXT / Markdown
- ✂️ 智能切分：按中文语义切块，控制 chunk 大小与重叠
- 🔍 向量检索：ChromaDB 本地持久化，无需额外数据库
- 🤖 大模型回答：OpenAI 兼容接口，DeepSeek / Qwen / 豆包 / OpenAI 随意切换
- 🔗 答案溯源：每条回答附带引用的原文片段
- 🖥 Web 界面：零构建、单页聊天界面，开箱即用

## 🧱 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI、Uvicorn |
| 大模型 | LangChain、LangChain-OpenAI |
| 向量库 | ChromaDB |
| 文档解析 | pypdf |
| 前端 | 原生 HTML + JS（无构建步骤） |

## 📁 目录结构

```
rag-knowledge-base/
├── app/
│   ├── main.py        # FastAPI 入口与 API 路由
│   ├── rag.py         # RAG 核心：切分 / 入库 / 检索 / 生成
│   ├── loader.py      # 文档解析（PDF / TXT / MD）
│   └── config.py      # 从 .env 读取配置
├── static/
│   └── index.html     # 单页聊天界面
├── scripts/
│   └── ingest.py      # 批量入库 data/ 目录
├── data/              # 示例文档
├── requirements.txt
└── .env.example
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

推荐使用 [SiliconFlow](https://siliconflow.cn)（国内可访问、便宜、OpenAI 兼容）：

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V3
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

打开 http://127.0.0.1:8000 即可使用。

启动时会打印一份**生效配置**，可用来确认 `.env` 是否真的读到了。

### 4. 批量入库示例文档（可选）

```bash
python scripts/ingest.py            # 增量入库，同名文档覆盖，不会产生重复
python scripts/ingest.py --reset    # 先清空向量库再全量入库
```

## 🔌 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务状态、模型信息、API Key 是否已配置 |
| POST | `/api/upload` | 上传文档（multipart file），同名文档会覆盖旧片段 |
| POST | `/api/ask` | 提问，返回答案与来源 |
| GET | `/api/count` | 已入库文档块数 |

### 提问示例

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "报销流程是什么？"}'
```

返回：

```json
{
  "answer": "报销流程为：第一步……",
  "sources": [
    { "content": "第一步：在 OA 填写报销单……", "source": "公司规章制度.md" }
  ]
}
```

## ⚙️ 可配置项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | 每块最大字数 |
| `CHUNK_OVERLAP` | 50 | 相邻块重叠字数，**必须小于 `CHUNK_SIZE`** |
| `TOP_K` | 4 | 检索返回条数，**必须 >= 1** |
| `REQUEST_TIMEOUT` | 60 | 单次模型请求超时（秒） |
| `MAX_UPLOAD_MB` | 20 | 单个上传文件大小上限 |
| `MAX_QUESTION_CHARS` | 2000 | 单个问题最大字数 |
| `CHROMA_DIR` | `./chroma_db` | 向量库目录，相对路径以**项目根目录**为基准 |

配置写错时，服务会在启动阶段直接给出中文提示并退出，不会等请求进来才报错。

## ⚠️ 部署注意事项

- **不要暴露到公网**：项目没有任何鉴权，任何人都能上传文档并消耗你的 API 额度。请保持 `--host 127.0.0.1`，或在前面加 Nginx + 认证。
- **改完 `.env` 必须重启进程**。`--reload` 只监听 `*.py`，改 `.env` 不会触发重启。
- **换 `EMBEDDING_MODEL` 后必须删除 `chroma_db` 重新入库**，否则旧向量维度对不上，检索会直接报错。
- **知识库文档是「不可信输入」**：系统提示词已声明忽略文档内的任何指令，但仍不建议上传来源不明的文档。
- 上传的文件应先自行确认是文本/PDF，二进制文件会被明确拒绝而不是静默变成乱码。

## 📝 工作原理

1. **加载** — 解析 PDF / TXT / MD，提取纯文本（校验类型与编码）
2. **切分** — `RecursiveCharacterTextSplitter` 按语义边界切块（500 字 / 重叠 50 字）
3. **向量化** — 调用 Embedding 模型生成向量，存入 ChromaDB
4. **检索** — 用户提问向量化后，检索最相关的 Top-K 片段
5. **生成** — 把检索到的上下文拼进 Prompt（用 `<资料>` 标签隔离），交给 LLM 生成带来源的回答

入库是**幂等**的：同一份文档重复上传/重复执行 `ingest.py` 不会产生重复片段。

## 🔜 后续计划

- [ ] 回答流式输出（SSE）
- [ ] 混合检索（向量 + 关键词）
- [ ] 引用溯源（标注命中原文段落）
- [ ] Docker 一键部署
- [ ] 对话历史与多轮追问
- [ ] 文档删除 / 知识库管理接口
