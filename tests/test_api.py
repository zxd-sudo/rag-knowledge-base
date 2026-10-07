"""HTTP 接口层回归测试。

重点覆盖「上传出错时返回的是友好 JSON，而不是裸 500 纯文本」这条修复 ——
旧版前端因此会把 "Unexpected token 'I'" 这种 JS 报错甩给用户。
"""
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _upload(client, filename: str, payload: bytes, content_type="application/octet-stream"):
    return client.post("/api/upload", files={"file": (filename, payload, content_type)})


# ---------- 健康检查 ----------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_model" in body
    assert "api_key_configured" in body


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "企业知识库问答" in resp.text


# ---------- 上传异常：必须是 JSON 400，而不是裸 500 ----------

def test_broken_pdf_returns_json_400_not_plain_500(client):
    resp = _upload(client, "broken.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()


def test_encrypted_pdf_returns_400(client):
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)

    resp = _upload(client, "enc.pdf", buf.getvalue(), "application/pdf")
    assert resp.status_code == 400
    assert "加密" in resp.json()["detail"]


def test_docx_returns_400(client):
    resp = _upload(client, "a.docx", b"PK\x03\x04\x14\x00", "application/octet-stream")
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


def test_binary_named_txt_returns_400(client):
    resp = _upload(client, "evil.txt", b"hello\x00world", "text/plain")
    assert resp.status_code == 400


def test_empty_file_returns_400(client):
    resp = _upload(client, "empty.txt", b"", "text/plain")
    assert resp.status_code == 400
    assert "空" in resp.json()["detail"]


def test_whitespace_only_document_returns_400(client):
    resp = _upload(client, "blank.txt", "   \n\n  ".encode(), "text/plain")
    assert resp.status_code == 400


# ---------- 上传正常路径 ----------

def test_upload_markdown_ok(client):
    resp = _upload(client, "doc.md", "# 标题\n\n正文内容。".encode(), "text/markdown")
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "doc.md"
    assert body["chunks"] >= 1


def test_upload_gbk_txt_ok_and_stored_correctly(client):
    resp = _upload(client, "gbk.txt", "中文编码测试".encode("gb18030"), "text/plain")
    assert resp.status_code == 200
    assert resp.json()["chunks"] == 1

    stored = client.get("/api/count").json()["documents"]
    assert stored == 1


def test_uploading_same_file_twice_does_not_grow_store(client):
    """旧版每上传一次就多一份副本。"""
    payload = "重复上传的内容。".encode()
    _upload(client, "dup.txt", payload, "text/plain")
    first = client.get("/api/count").json()["documents"]

    _upload(client, "dup.txt", payload, "text/plain")
    _upload(client, "dup.txt", payload, "text/plain")
    assert client.get("/api/count").json()["documents"] == first


def test_oversize_upload_returns_413(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    resp = _upload(client, "big.txt", b"x" * (2 * 1024 * 1024), "text/plain")
    assert resp.status_code == 413
    assert "上限" in resp.json()["detail"]


# ---------- 提问 ----------

def test_ask_ok(client):
    _upload(client, "rule.md", "报销流程：第一步在 OA 填单。".encode(), "text/markdown")
    resp = client.post("/api/ask", json={"question": "报销流程？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "测试回答"
    assert isinstance(body["sources"], list)


def test_ask_empty_question_returns_400(client):
    resp = client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_missing_field_returns_422(client):
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422


def test_ask_too_long_question_returns_422(client):
    """旧版没有长度上限，100 万字符的问题会直接打到模型上。"""
    resp = client.post("/api/ask", json={"question": "x" * (config.MAX_QUESTION_CHARS + 100)})
    assert resp.status_code == 422


# ---------- 计数接口 ----------

def test_count_on_empty_store(client):
    resp = client.get("/api/count")
    assert resp.status_code == 200
    assert resp.json()["documents"] == 0
