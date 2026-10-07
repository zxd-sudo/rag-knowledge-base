"""文档解析的回归测试：类型白名单、编码识别、加密 PDF。"""
import io

import pytest
from pypdf import PdfWriter
from pypdf.errors import PdfStreamError

from app.loader import UnsupportedFileError, decode_text, parse_bytes


def _encrypted_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------- 正常路径 ----------

def test_markdown_passthrough():
    text = "# 标题\n\n正文内容。"
    assert parse_bytes(text.encode("utf-8"), "doc.md") == text


def test_txt_ok():
    assert parse_bytes("纯文本".encode("utf-8"), "a.txt") == "纯文本"


def test_utf8_bom_is_stripped():
    """BOM 不能混进第一个分块。"""
    assert decode_text("内容".encode("utf-8-sig")) == "内容"


def test_gbk_is_decoded_not_mojibake():
    """GBK 文本要正确解码，不能变成乱码（旧版用 errors=ignore 会静默丢字）。"""
    assert decode_text("中文编码测试".encode("gb18030")) == "中文编码测试"


# ---------- 类型白名单 ----------

@pytest.mark.parametrize("name", ["a.docx", "a.doc", "a.xlsx", "a.png", "a.zip", "noext"])
def test_unsupported_suffix_is_rejected(name):
    with pytest.raises(UnsupportedFileError):
        parse_bytes(b"anything", name)


@pytest.mark.parametrize(
    "payload",
    [
        b"PK\x03\x04\x14\x00\x00\x00",      # docx/xlsx 其实是个 zip
        b"\x89PNG\r\n\x1a\n\x00\x00",       # PNG
        b"hello\x00world",                   # 含 NUL
        bytes(range(1, 32)) * 40,            # 控制字符占比过高
    ],
)
def test_binary_content_rejected_even_when_named_txt(payload):
    """伪装成 .txt 的二进制文件必须被拒绝，而不是被硬解码成乱码入库。"""
    with pytest.raises(UnsupportedFileError):
        parse_bytes(payload, "evil.txt")


# ---------- PDF 异常路径 ----------

def test_broken_pdf_raises_pdf_error():
    """损坏的 PDF 会抛 pypdf 异常，由接口层转成 400（这里只验证它确实抛出）。"""
    with pytest.raises(PdfStreamError):
        parse_bytes(b"%PDF-1.4 this is not a real pdf", "broken.pdf")


def test_encrypted_pdf_is_rejected_with_clear_message():
    with pytest.raises(UnsupportedFileError, match="加密"):
        parse_bytes(_encrypted_pdf_bytes(), "enc.pdf")
