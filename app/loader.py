"""把上传的文档转成纯文本。支持 PDF / TXT / Markdown。

这里做两件以前没做的事：
1. 白名单校验文件类型，二进制文件不再被硬解码成乱码塞进知识库；
2. 编码识别失败就明确报错，不再用 errors="ignore" 静默丢字符。
"""
import io
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# 常见二进制格式的文件头。后缀不符时直接判定“传错文件”，而不是硬解码
_BINARY_MAGIC = (
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",   # zip / docx / xlsx / pptx
    b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"BM",     # 图片
    b"\xd0\xcf\x11\xe0",                             # 老版 doc / xls / ppt
    b"\x1f\x8b",                                     # gzip
    b"%PDF",                                         # 后缀不是 .pdf 的 PDF
    b"RIFF", b"OggS", b"ID3", b"\x00\x00\x01\x00",
    b"\x7fELF", b"SQLite format 3",
)


class UnsupportedFileError(ValueError):
    """文件类型或编码不受支持 —— 属于用户输入问题，接口应返回 400。"""


def _looks_binary(data: bytes) -> bool:
    head = data[:16]
    if any(head.startswith(magic) for magic in _BINARY_MAGIC):
        return True
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    # 控制字符占比过高也判为二进制（排除制表符/换行/回车）
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    return bool(sample) and control / len(sample) > 0.05


def decode_text(data: bytes) -> str:
    """按 UTF-8 → GB18030 → Big5 依次尝试；全部失败则报错。"""
    if _looks_binary(data):
        raise UnsupportedFileError(
            "这看起来是二进制文件（例如 Word / Excel / 图片），不是纯文本。"
            "请上传 PDF / TXT / Markdown，或先把文档另存为 PDF。"
        )
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            # utf-8-sig 会顺手去掉 BOM，避免 BOM 混进第一个分块
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedFileError("无法识别文件编码，请先把文档转成 UTF-8 再上传。")


def _parse_pdf(stream: io.BytesIO) -> str:
    from pypdf import PdfReader

    reader = PdfReader(stream)
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # 有些 PDF 只是限制权限，空密码可以解
        except Exception:  # noqa: BLE001
            pass
        if reader.is_encrypted:
            raise UnsupportedFileError("该 PDF 已加密，请先解密或另存为未加密版本。")
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_bytes(data: bytes, filename: str = "") -> str:
    """把字节流解析成纯文本。类型不支持时抛 UnsupportedFileError。"""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_SUFFIXES))
        raise UnsupportedFileError(
            f"不支持的文件类型：{suffix or '(无扩展名)'}。仅支持 {allowed}"
        )
    if suffix == ".pdf":
        return _parse_pdf(io.BytesIO(data))
    return decode_text(data)


def load_file(path: Path) -> str:
    """给 scripts/ingest.py 用：从磁盘读取并解析。"""
    return parse_bytes(path.read_bytes(), path.name)
