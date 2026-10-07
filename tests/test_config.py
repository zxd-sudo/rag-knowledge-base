"""配置校验的回归测试。

配置是在 import 阶段求值的，所以必须在子进程里跑，才能验证「启动即报错」的行为。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 这些变量会被子进程继承，必须先清掉，避免受外部环境影响
_MANAGED = (
    "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "CHROMA_DIR",
    "MAX_UPLOAD_MB", "MAX_QUESTION_CHARS", "REQUEST_TIMEOUT",
)

_CODE = (
    "from app import config; "
    "print('OK', config.CHUNK_SIZE, config.TOP_K, config.CHROMA_DIR)"
)


def _run(env_extra: dict, cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key in _MANAGED:
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("OPENAI_API_KEY", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output=True, text=True, env=env, cwd=str(cwd), timeout=60,
    )


def _output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


# ---------- 正常路径 ----------

def test_defaults_are_valid(tmp_path):
    result = _run({}, tmp_path)
    assert result.returncode == 0, _output(result)
    assert "OK 500 4" in result.stdout


# ---------- 非法数值必须启动即报错，而不是等请求进来才崩 ----------

def test_non_integer_value_fails_fast(tmp_path):
    result = _run({"CHUNK_SIZE": "abc"}, tmp_path)
    assert result.returncode != 0
    assert "不是整数" in _output(result)
    assert "CHUNK_SIZE" in _output(result)


def test_empty_value_fails_fast(tmp_path):
    result = _run({"CHUNK_SIZE": ""}, tmp_path)
    # 空字符串按「未设置」处理，回退默认值，属于合理行为
    assert result.returncode == 0
    assert "OK 500" in result.stdout


def test_top_k_zero_is_rejected(tmp_path):
    """旧版能通过校验，然后在 similarity_search 里抛 TypeError。"""
    result = _run({"TOP_K": "0"}, tmp_path)
    assert result.returncode != 0
    assert "必须 >= 1" in _output(result)


def test_top_k_negative_is_rejected(tmp_path):
    result = _run({"TOP_K": "-3"}, tmp_path)
    assert result.returncode != 0
    assert "必须 >= 1" in _output(result)


def test_overlap_not_smaller_than_chunk_size_is_rejected(tmp_path):
    """旧版直到切分文档时才抛 ValueError。"""
    result = _run({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, tmp_path)
    assert result.returncode != 0
    assert "必须小于" in _output(result)


def test_error_message_is_chinese_and_actionable(tmp_path):
    out = _output(_run({"TOP_K": "abc"}, tmp_path))
    assert "❌" in out and ".env" in out


# ---------- 路径解析必须与启动目录无关 ----------

def test_chroma_dir_is_resolved_against_project_root(tmp_path):
    """从别的目录启动，向量库也必须落在项目里（旧版会落到 CWD）。"""
    result = _run({}, tmp_path)
    assert result.returncode == 0, _output(result)
    expected = str(ROOT / "chroma_db")
    assert expected in result.stdout, f"期望包含 {expected}，实际输出：{result.stdout}"


def test_absolute_chroma_dir_is_kept(tmp_path):
    custom = tmp_path / "my_chroma"
    result = _run({"CHROMA_DIR": str(custom)}, tmp_path)
    assert result.returncode == 0, _output(result)
    assert str(custom) in result.stdout
