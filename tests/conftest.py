"""测试公共夹具。

两个原则：
1. **不联网**：假的 embedding / 假的 LLM，CI 里不需要任何 API Key；
2. **不碰真实数据**：向量库一律指向临时目录，绝不写项目里的 chroma_db。

两个夹具都是 autouse，所以任何测试都不可能「不小心」打到真实服务。
"""
import hashlib
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, rag  # noqa: E402


class FakeEmbeddings(Embeddings):
    """确定性的假向量：同一个文本永远得到同一个向量。"""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def isolated_vectorstore(tmp_path, monkeypatch):
    """把向量库目录换成临时目录，并用假 embedding 顶掉真实模型。"""
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag, "get_embeddings", lambda: FakeEmbeddings())
    rag.get_vectorstore.cache_clear()
    yield
    rag.get_vectorstore.cache_clear()


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """顶掉真实大模型，任何测试都不会发起网络请求。"""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(
        rag,
        "get_llm",
        lambda: RunnableLambda(lambda _: AIMessage(content="测试回答")),
    )
