"""把 data/ 目录下的文档批量入库。

用法：
    python scripts/ingest.py            # 增量入库（同名文档覆盖，不会产生重复）
    python scripts/ingest.py --reset    # 先清空向量库再入库
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.loader import UnsupportedFileError, load_file  # noqa: E402
from app.rag import count_documents, get_vectorstore, ingest_text  # noqa: E402

DATA_DIR = config.BASE_DIR / "data"
SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


def main() -> int:
    reset = "--reset" in sys.argv[1:]

    print("📋 生效配置：")
    print("   " + config.summary().replace("\n", "\n   "))
    print()

    if reset:
        get_vectorstore().delete_collection()
        print("🧹 已清空向量库\n")

    files = [f for f in sorted(DATA_DIR.glob("*")) if f.suffix.lower() in SUFFIXES]
    if not files:
        print(f"⚠️ {DATA_DIR} 里没有可入库的文档（{'/'.join(SUFFIXES)}）")
        return 1

    total, failed = 0, 0
    for path in files:
        # 单个文件出错不再中断整批
        try:
            text = load_file(path)
        except UnsupportedFileError as e:
            print(f"⏭️  跳过 {path.name}：{e}")
            failed += 1
            continue
        except Exception as e:  # noqa: BLE001
            print(f"❌ {path.name} 解析失败：{e}")
            failed += 1
            continue

        if not text.strip():
            print(f"⏭️  跳过 {path.name}：解析后内容为空")
            continue

        try:
            chunks = ingest_text(text, metadata={"source": path.name})
        except Exception as e:  # noqa: BLE001
            print(f"❌ {path.name} 入库失败：{e}")
            failed += 1
            continue

        total += chunks
        print(f"✅ {path.name} -> {chunks} 块")

    print(f"\n本次入库 {total} 块，失败 {failed} 个文件，向量库现有 {count_documents()} 块。")
    if failed:
        print("⚠️ 有文件没能入库，请检查上面的报错。")
    print("现在可以启动服务提问了：uvicorn app.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
