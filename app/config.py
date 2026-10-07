"""从 .env 读取配置，全部集中在这一处。

三个关键约定：
1. .env 固定从项目根目录读取，不再依赖启动时的工作目录；
2. 相对路径（如 CHROMA_DIR）一律相对项目根目录解析；
3. 数值配置在启动时校验，写错立刻给出中文提示，而不是等请求进来才崩。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app/ 的上一级），所有相对路径都以它为基准
BASE_DIR = Path(__file__).resolve().parent.parent

# 显式指定路径，避免“在别的目录启动就找不到 .env”
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """读整数配置；写错立刻退出并说明原因，而不是留到运行时抛 ValueError。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise SystemExit(
            f"❌ .env 配置错误：{name}={raw!r} 不是整数，请改成数字（默认 {default}）。"
        ) from None
    if minimum is not None and value < minimum:
        raise SystemExit(f"❌ .env 配置错误：{name}={value}，必须 >= {minimum}。")
    return value


API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

CHUNK_SIZE = _env_int("CHUNK_SIZE", 500, minimum=1)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50, minimum=0)
TOP_K = _env_int("TOP_K", 4, minimum=1)

# 请求层面的硬限制，防止单个请求打爆内存或账单
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 20, minimum=1)
MAX_QUESTION_CHARS = _env_int("MAX_QUESTION_CHARS", 2000, minimum=1)
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 60, minimum=1)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise SystemExit(
        f"❌ .env 配置错误：CHUNK_OVERLAP({CHUNK_OVERLAP}) 必须小于 "
        f"CHUNK_SIZE({CHUNK_SIZE})。"
    )


def _resolve_path(raw: str) -> str:
    """相对路径基于项目根目录解析，而不是当前工作目录。"""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p)


CHROMA_DIR = _resolve_path(os.getenv("CHROMA_DIR", "./chroma_db").strip() or "./chroma_db")

# 向量库集合名。注意：换 EMBEDDING_MODEL 后维度会变，
# 必须删除 chroma_db 重新入库，否则检索会报维度不一致。
COLLECTION_NAME = "knowledge_base"


def summary() -> str:
    """启动时打印，用于确认 .env 到底有没有生效（.env 改动必须重启进程）。"""
    key = "已配置" if API_KEY else "⚠️ 未配置（提问会失败）"
    return (
        f"LLM={LLM_MODEL} | Embedding={EMBEDDING_MODEL} | API Key={key}\n"
        f"BASE_URL={BASE_URL}\n"
        f"CHROMA_DIR={CHROMA_DIR}\n"
        f"CHUNK_SIZE={CHUNK_SIZE} CHUNK_OVERLAP={CHUNK_OVERLAP} TOP_K={TOP_K}\n"
        f"REQUEST_TIMEOUT={REQUEST_TIMEOUT}s 最大上传={MAX_UPLOAD_MB}MB "
        f"问题最长={MAX_QUESTION_CHARS}字"
    )
