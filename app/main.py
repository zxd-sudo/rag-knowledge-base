"""FastAPI 入口：上传文档、提问、静态页面。"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .loader import UnsupportedFileError, parse_bytes
from .rag import ask, count_documents, ingest_text

logger = logging.getLogger("rag")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时把生效配置打出来 —— .env 改完必须重启进程才会生效。"""
    print("📋 生效配置（修改 .env 后需重启进程）：")
    print("   " + config.summary().replace("\n", "\n   "))
    if not config.API_KEY:
        print("   ⚠️ OPENAI_API_KEY 未配置，提问会失败。请编辑 "
              f"{BASE_DIR / '.env'}")
    yield


app = FastAPI(title="企业知识库问答", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., max_length=config.MAX_QUESTION_CHARS)


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        # 让前端能区分「服务活着」和「Key 配好了」
        "api_key_configured": bool(config.API_KEY),
    }


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """同步端点：FastAPI 会自动放进线程池，不会阻塞事件循环。"""
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    name = file.filename or ""
    data = file.file.read(limit + 1)

    if not data:
        raise HTTPException(400, "文件是空的")
    if len(data) > limit:
        raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB} MB 上限，请拆分后再上传")

    try:
        text = parse_bytes(data, name)
    except UnsupportedFileError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # 损坏的 PDF 等 pypdf 异常
        logger.exception("解析文档失败：%s", name)
        raise HTTPException(400, f"文档解析失败：{e}") from e

    if not text.strip():
        raise HTTPException(400, "文档解析后为空（扫描版 PDF 需要先做 OCR）")

    try:
        chunks = ingest_text(text, metadata={"source": name or "未命名文档"})
    except Exception as e:  # noqa: BLE001
        logger.exception("入库失败：%s", name)
        raise HTTPException(500, f"写入向量库失败：{e}") from e

    return {"filename": name, "chunks": chunks}


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")

    try:
        return ask(question)
    except openai.AuthenticationError as e:
        raise HTTPException(
            401, "API Key 无效或已过期，请检查 .env 里的 OPENAI_API_KEY"
        ) from e
    except openai.APIConnectionError as e:
        raise HTTPException(
            502, f"连不上模型接口 {config.BASE_URL}，请检查网络或代理设置"
        ) from e
    except openai.APIError as e:
        raise HTTPException(502, f"模型接口返回错误：{e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("问答失败")
        raise HTTPException(500, f"服务内部错误：{e}") from e


@app.get("/api/count")
def count():
    try:
        return {"documents": count_documents()}
    except Exception as e:  # noqa: BLE001
        logger.exception("读取向量库失败")
        raise HTTPException(500, f"读取向量库失败：{e}") from e
